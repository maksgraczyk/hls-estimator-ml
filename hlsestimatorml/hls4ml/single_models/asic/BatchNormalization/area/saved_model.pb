ñ«9
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b685
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
dense_711/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:i*!
shared_namedense_711/kernel
u
$dense_711/kernel/Read/ReadVariableOpReadVariableOpdense_711/kernel*
_output_shapes

:i*
dtype0
t
dense_711/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*
shared_namedense_711/bias
m
"dense_711/bias/Read/ReadVariableOpReadVariableOpdense_711/bias*
_output_shapes
:i*
dtype0

batch_normalization_640/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*.
shared_namebatch_normalization_640/gamma

1batch_normalization_640/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_640/gamma*
_output_shapes
:i*
dtype0

batch_normalization_640/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*-
shared_namebatch_normalization_640/beta

0batch_normalization_640/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_640/beta*
_output_shapes
:i*
dtype0

#batch_normalization_640/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*4
shared_name%#batch_normalization_640/moving_mean

7batch_normalization_640/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_640/moving_mean*
_output_shapes
:i*
dtype0
¦
'batch_normalization_640/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*8
shared_name)'batch_normalization_640/moving_variance

;batch_normalization_640/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_640/moving_variance*
_output_shapes
:i*
dtype0
|
dense_712/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ii*!
shared_namedense_712/kernel
u
$dense_712/kernel/Read/ReadVariableOpReadVariableOpdense_712/kernel*
_output_shapes

:ii*
dtype0
t
dense_712/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*
shared_namedense_712/bias
m
"dense_712/bias/Read/ReadVariableOpReadVariableOpdense_712/bias*
_output_shapes
:i*
dtype0

batch_normalization_641/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*.
shared_namebatch_normalization_641/gamma

1batch_normalization_641/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_641/gamma*
_output_shapes
:i*
dtype0

batch_normalization_641/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*-
shared_namebatch_normalization_641/beta

0batch_normalization_641/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_641/beta*
_output_shapes
:i*
dtype0

#batch_normalization_641/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*4
shared_name%#batch_normalization_641/moving_mean

7batch_normalization_641/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_641/moving_mean*
_output_shapes
:i*
dtype0
¦
'batch_normalization_641/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*8
shared_name)'batch_normalization_641/moving_variance

;batch_normalization_641/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_641/moving_variance*
_output_shapes
:i*
dtype0
|
dense_713/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ii*!
shared_namedense_713/kernel
u
$dense_713/kernel/Read/ReadVariableOpReadVariableOpdense_713/kernel*
_output_shapes

:ii*
dtype0
t
dense_713/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*
shared_namedense_713/bias
m
"dense_713/bias/Read/ReadVariableOpReadVariableOpdense_713/bias*
_output_shapes
:i*
dtype0

batch_normalization_642/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*.
shared_namebatch_normalization_642/gamma

1batch_normalization_642/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_642/gamma*
_output_shapes
:i*
dtype0

batch_normalization_642/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*-
shared_namebatch_normalization_642/beta

0batch_normalization_642/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_642/beta*
_output_shapes
:i*
dtype0

#batch_normalization_642/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*4
shared_name%#batch_normalization_642/moving_mean

7batch_normalization_642/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_642/moving_mean*
_output_shapes
:i*
dtype0
¦
'batch_normalization_642/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*8
shared_name)'batch_normalization_642/moving_variance

;batch_normalization_642/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_642/moving_variance*
_output_shapes
:i*
dtype0
|
dense_714/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ii*!
shared_namedense_714/kernel
u
$dense_714/kernel/Read/ReadVariableOpReadVariableOpdense_714/kernel*
_output_shapes

:ii*
dtype0
t
dense_714/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*
shared_namedense_714/bias
m
"dense_714/bias/Read/ReadVariableOpReadVariableOpdense_714/bias*
_output_shapes
:i*
dtype0

batch_normalization_643/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*.
shared_namebatch_normalization_643/gamma

1batch_normalization_643/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_643/gamma*
_output_shapes
:i*
dtype0

batch_normalization_643/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*-
shared_namebatch_normalization_643/beta

0batch_normalization_643/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_643/beta*
_output_shapes
:i*
dtype0

#batch_normalization_643/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*4
shared_name%#batch_normalization_643/moving_mean

7batch_normalization_643/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_643/moving_mean*
_output_shapes
:i*
dtype0
¦
'batch_normalization_643/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*8
shared_name)'batch_normalization_643/moving_variance

;batch_normalization_643/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_643/moving_variance*
_output_shapes
:i*
dtype0
|
dense_715/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ii*!
shared_namedense_715/kernel
u
$dense_715/kernel/Read/ReadVariableOpReadVariableOpdense_715/kernel*
_output_shapes

:ii*
dtype0
t
dense_715/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*
shared_namedense_715/bias
m
"dense_715/bias/Read/ReadVariableOpReadVariableOpdense_715/bias*
_output_shapes
:i*
dtype0

batch_normalization_644/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*.
shared_namebatch_normalization_644/gamma

1batch_normalization_644/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_644/gamma*
_output_shapes
:i*
dtype0

batch_normalization_644/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*-
shared_namebatch_normalization_644/beta

0batch_normalization_644/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_644/beta*
_output_shapes
:i*
dtype0

#batch_normalization_644/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*4
shared_name%#batch_normalization_644/moving_mean

7batch_normalization_644/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_644/moving_mean*
_output_shapes
:i*
dtype0
¦
'batch_normalization_644/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*8
shared_name)'batch_normalization_644/moving_variance

;batch_normalization_644/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_644/moving_variance*
_output_shapes
:i*
dtype0
|
dense_716/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:i=*!
shared_namedense_716/kernel
u
$dense_716/kernel/Read/ReadVariableOpReadVariableOpdense_716/kernel*
_output_shapes

:i=*
dtype0
t
dense_716/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*
shared_namedense_716/bias
m
"dense_716/bias/Read/ReadVariableOpReadVariableOpdense_716/bias*
_output_shapes
:=*
dtype0

batch_normalization_645/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*.
shared_namebatch_normalization_645/gamma

1batch_normalization_645/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_645/gamma*
_output_shapes
:=*
dtype0

batch_normalization_645/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*-
shared_namebatch_normalization_645/beta

0batch_normalization_645/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_645/beta*
_output_shapes
:=*
dtype0

#batch_normalization_645/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*4
shared_name%#batch_normalization_645/moving_mean

7batch_normalization_645/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_645/moving_mean*
_output_shapes
:=*
dtype0
¦
'batch_normalization_645/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*8
shared_name)'batch_normalization_645/moving_variance

;batch_normalization_645/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_645/moving_variance*
_output_shapes
:=*
dtype0
|
dense_717/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:==*!
shared_namedense_717/kernel
u
$dense_717/kernel/Read/ReadVariableOpReadVariableOpdense_717/kernel*
_output_shapes

:==*
dtype0
t
dense_717/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*
shared_namedense_717/bias
m
"dense_717/bias/Read/ReadVariableOpReadVariableOpdense_717/bias*
_output_shapes
:=*
dtype0

batch_normalization_646/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*.
shared_namebatch_normalization_646/gamma

1batch_normalization_646/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_646/gamma*
_output_shapes
:=*
dtype0

batch_normalization_646/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*-
shared_namebatch_normalization_646/beta

0batch_normalization_646/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_646/beta*
_output_shapes
:=*
dtype0

#batch_normalization_646/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*4
shared_name%#batch_normalization_646/moving_mean

7batch_normalization_646/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_646/moving_mean*
_output_shapes
:=*
dtype0
¦
'batch_normalization_646/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*8
shared_name)'batch_normalization_646/moving_variance

;batch_normalization_646/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_646/moving_variance*
_output_shapes
:=*
dtype0
|
dense_718/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:=7*!
shared_namedense_718/kernel
u
$dense_718/kernel/Read/ReadVariableOpReadVariableOpdense_718/kernel*
_output_shapes

:=7*
dtype0
t
dense_718/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*
shared_namedense_718/bias
m
"dense_718/bias/Read/ReadVariableOpReadVariableOpdense_718/bias*
_output_shapes
:7*
dtype0

batch_normalization_647/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*.
shared_namebatch_normalization_647/gamma

1batch_normalization_647/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_647/gamma*
_output_shapes
:7*
dtype0

batch_normalization_647/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*-
shared_namebatch_normalization_647/beta

0batch_normalization_647/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_647/beta*
_output_shapes
:7*
dtype0

#batch_normalization_647/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*4
shared_name%#batch_normalization_647/moving_mean

7batch_normalization_647/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_647/moving_mean*
_output_shapes
:7*
dtype0
¦
'batch_normalization_647/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*8
shared_name)'batch_normalization_647/moving_variance

;batch_normalization_647/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_647/moving_variance*
_output_shapes
:7*
dtype0
|
dense_719/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:77*!
shared_namedense_719/kernel
u
$dense_719/kernel/Read/ReadVariableOpReadVariableOpdense_719/kernel*
_output_shapes

:77*
dtype0
t
dense_719/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*
shared_namedense_719/bias
m
"dense_719/bias/Read/ReadVariableOpReadVariableOpdense_719/bias*
_output_shapes
:7*
dtype0

batch_normalization_648/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*.
shared_namebatch_normalization_648/gamma

1batch_normalization_648/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_648/gamma*
_output_shapes
:7*
dtype0

batch_normalization_648/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*-
shared_namebatch_normalization_648/beta

0batch_normalization_648/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_648/beta*
_output_shapes
:7*
dtype0

#batch_normalization_648/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*4
shared_name%#batch_normalization_648/moving_mean

7batch_normalization_648/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_648/moving_mean*
_output_shapes
:7*
dtype0
¦
'batch_normalization_648/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*8
shared_name)'batch_normalization_648/moving_variance

;batch_normalization_648/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_648/moving_variance*
_output_shapes
:7*
dtype0
|
dense_720/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7*!
shared_namedense_720/kernel
u
$dense_720/kernel/Read/ReadVariableOpReadVariableOpdense_720/kernel*
_output_shapes

:7*
dtype0
t
dense_720/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_720/bias
m
"dense_720/bias/Read/ReadVariableOpReadVariableOpdense_720/bias*
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
Adam/dense_711/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:i*(
shared_nameAdam/dense_711/kernel/m

+Adam/dense_711/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_711/kernel/m*
_output_shapes

:i*
dtype0

Adam/dense_711/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*&
shared_nameAdam/dense_711/bias/m
{
)Adam/dense_711/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_711/bias/m*
_output_shapes
:i*
dtype0
 
$Adam/batch_normalization_640/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*5
shared_name&$Adam/batch_normalization_640/gamma/m

8Adam/batch_normalization_640/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_640/gamma/m*
_output_shapes
:i*
dtype0

#Adam/batch_normalization_640/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*4
shared_name%#Adam/batch_normalization_640/beta/m

7Adam/batch_normalization_640/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_640/beta/m*
_output_shapes
:i*
dtype0

Adam/dense_712/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ii*(
shared_nameAdam/dense_712/kernel/m

+Adam/dense_712/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_712/kernel/m*
_output_shapes

:ii*
dtype0

Adam/dense_712/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*&
shared_nameAdam/dense_712/bias/m
{
)Adam/dense_712/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_712/bias/m*
_output_shapes
:i*
dtype0
 
$Adam/batch_normalization_641/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*5
shared_name&$Adam/batch_normalization_641/gamma/m

8Adam/batch_normalization_641/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_641/gamma/m*
_output_shapes
:i*
dtype0

#Adam/batch_normalization_641/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*4
shared_name%#Adam/batch_normalization_641/beta/m

7Adam/batch_normalization_641/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_641/beta/m*
_output_shapes
:i*
dtype0

Adam/dense_713/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ii*(
shared_nameAdam/dense_713/kernel/m

+Adam/dense_713/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_713/kernel/m*
_output_shapes

:ii*
dtype0

Adam/dense_713/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*&
shared_nameAdam/dense_713/bias/m
{
)Adam/dense_713/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_713/bias/m*
_output_shapes
:i*
dtype0
 
$Adam/batch_normalization_642/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*5
shared_name&$Adam/batch_normalization_642/gamma/m

8Adam/batch_normalization_642/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_642/gamma/m*
_output_shapes
:i*
dtype0

#Adam/batch_normalization_642/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*4
shared_name%#Adam/batch_normalization_642/beta/m

7Adam/batch_normalization_642/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_642/beta/m*
_output_shapes
:i*
dtype0

Adam/dense_714/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ii*(
shared_nameAdam/dense_714/kernel/m

+Adam/dense_714/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_714/kernel/m*
_output_shapes

:ii*
dtype0

Adam/dense_714/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*&
shared_nameAdam/dense_714/bias/m
{
)Adam/dense_714/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_714/bias/m*
_output_shapes
:i*
dtype0
 
$Adam/batch_normalization_643/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*5
shared_name&$Adam/batch_normalization_643/gamma/m

8Adam/batch_normalization_643/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_643/gamma/m*
_output_shapes
:i*
dtype0

#Adam/batch_normalization_643/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*4
shared_name%#Adam/batch_normalization_643/beta/m

7Adam/batch_normalization_643/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_643/beta/m*
_output_shapes
:i*
dtype0

Adam/dense_715/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ii*(
shared_nameAdam/dense_715/kernel/m

+Adam/dense_715/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_715/kernel/m*
_output_shapes

:ii*
dtype0

Adam/dense_715/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*&
shared_nameAdam/dense_715/bias/m
{
)Adam/dense_715/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_715/bias/m*
_output_shapes
:i*
dtype0
 
$Adam/batch_normalization_644/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*5
shared_name&$Adam/batch_normalization_644/gamma/m

8Adam/batch_normalization_644/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_644/gamma/m*
_output_shapes
:i*
dtype0

#Adam/batch_normalization_644/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*4
shared_name%#Adam/batch_normalization_644/beta/m

7Adam/batch_normalization_644/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_644/beta/m*
_output_shapes
:i*
dtype0

Adam/dense_716/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:i=*(
shared_nameAdam/dense_716/kernel/m

+Adam/dense_716/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_716/kernel/m*
_output_shapes

:i=*
dtype0

Adam/dense_716/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*&
shared_nameAdam/dense_716/bias/m
{
)Adam/dense_716/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_716/bias/m*
_output_shapes
:=*
dtype0
 
$Adam/batch_normalization_645/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*5
shared_name&$Adam/batch_normalization_645/gamma/m

8Adam/batch_normalization_645/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_645/gamma/m*
_output_shapes
:=*
dtype0

#Adam/batch_normalization_645/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*4
shared_name%#Adam/batch_normalization_645/beta/m

7Adam/batch_normalization_645/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_645/beta/m*
_output_shapes
:=*
dtype0

Adam/dense_717/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:==*(
shared_nameAdam/dense_717/kernel/m

+Adam/dense_717/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_717/kernel/m*
_output_shapes

:==*
dtype0

Adam/dense_717/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*&
shared_nameAdam/dense_717/bias/m
{
)Adam/dense_717/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_717/bias/m*
_output_shapes
:=*
dtype0
 
$Adam/batch_normalization_646/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*5
shared_name&$Adam/batch_normalization_646/gamma/m

8Adam/batch_normalization_646/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_646/gamma/m*
_output_shapes
:=*
dtype0

#Adam/batch_normalization_646/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*4
shared_name%#Adam/batch_normalization_646/beta/m

7Adam/batch_normalization_646/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_646/beta/m*
_output_shapes
:=*
dtype0

Adam/dense_718/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:=7*(
shared_nameAdam/dense_718/kernel/m

+Adam/dense_718/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_718/kernel/m*
_output_shapes

:=7*
dtype0

Adam/dense_718/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*&
shared_nameAdam/dense_718/bias/m
{
)Adam/dense_718/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_718/bias/m*
_output_shapes
:7*
dtype0
 
$Adam/batch_normalization_647/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*5
shared_name&$Adam/batch_normalization_647/gamma/m

8Adam/batch_normalization_647/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_647/gamma/m*
_output_shapes
:7*
dtype0

#Adam/batch_normalization_647/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*4
shared_name%#Adam/batch_normalization_647/beta/m

7Adam/batch_normalization_647/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_647/beta/m*
_output_shapes
:7*
dtype0

Adam/dense_719/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:77*(
shared_nameAdam/dense_719/kernel/m

+Adam/dense_719/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_719/kernel/m*
_output_shapes

:77*
dtype0

Adam/dense_719/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*&
shared_nameAdam/dense_719/bias/m
{
)Adam/dense_719/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_719/bias/m*
_output_shapes
:7*
dtype0
 
$Adam/batch_normalization_648/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*5
shared_name&$Adam/batch_normalization_648/gamma/m

8Adam/batch_normalization_648/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_648/gamma/m*
_output_shapes
:7*
dtype0

#Adam/batch_normalization_648/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*4
shared_name%#Adam/batch_normalization_648/beta/m

7Adam/batch_normalization_648/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_648/beta/m*
_output_shapes
:7*
dtype0

Adam/dense_720/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7*(
shared_nameAdam/dense_720/kernel/m

+Adam/dense_720/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_720/kernel/m*
_output_shapes

:7*
dtype0

Adam/dense_720/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_720/bias/m
{
)Adam/dense_720/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_720/bias/m*
_output_shapes
:*
dtype0

Adam/dense_711/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:i*(
shared_nameAdam/dense_711/kernel/v

+Adam/dense_711/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_711/kernel/v*
_output_shapes

:i*
dtype0

Adam/dense_711/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*&
shared_nameAdam/dense_711/bias/v
{
)Adam/dense_711/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_711/bias/v*
_output_shapes
:i*
dtype0
 
$Adam/batch_normalization_640/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*5
shared_name&$Adam/batch_normalization_640/gamma/v

8Adam/batch_normalization_640/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_640/gamma/v*
_output_shapes
:i*
dtype0

#Adam/batch_normalization_640/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*4
shared_name%#Adam/batch_normalization_640/beta/v

7Adam/batch_normalization_640/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_640/beta/v*
_output_shapes
:i*
dtype0

Adam/dense_712/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ii*(
shared_nameAdam/dense_712/kernel/v

+Adam/dense_712/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_712/kernel/v*
_output_shapes

:ii*
dtype0

Adam/dense_712/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*&
shared_nameAdam/dense_712/bias/v
{
)Adam/dense_712/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_712/bias/v*
_output_shapes
:i*
dtype0
 
$Adam/batch_normalization_641/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*5
shared_name&$Adam/batch_normalization_641/gamma/v

8Adam/batch_normalization_641/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_641/gamma/v*
_output_shapes
:i*
dtype0

#Adam/batch_normalization_641/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*4
shared_name%#Adam/batch_normalization_641/beta/v

7Adam/batch_normalization_641/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_641/beta/v*
_output_shapes
:i*
dtype0

Adam/dense_713/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ii*(
shared_nameAdam/dense_713/kernel/v

+Adam/dense_713/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_713/kernel/v*
_output_shapes

:ii*
dtype0

Adam/dense_713/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*&
shared_nameAdam/dense_713/bias/v
{
)Adam/dense_713/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_713/bias/v*
_output_shapes
:i*
dtype0
 
$Adam/batch_normalization_642/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*5
shared_name&$Adam/batch_normalization_642/gamma/v

8Adam/batch_normalization_642/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_642/gamma/v*
_output_shapes
:i*
dtype0

#Adam/batch_normalization_642/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*4
shared_name%#Adam/batch_normalization_642/beta/v

7Adam/batch_normalization_642/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_642/beta/v*
_output_shapes
:i*
dtype0

Adam/dense_714/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ii*(
shared_nameAdam/dense_714/kernel/v

+Adam/dense_714/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_714/kernel/v*
_output_shapes

:ii*
dtype0

Adam/dense_714/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*&
shared_nameAdam/dense_714/bias/v
{
)Adam/dense_714/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_714/bias/v*
_output_shapes
:i*
dtype0
 
$Adam/batch_normalization_643/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*5
shared_name&$Adam/batch_normalization_643/gamma/v

8Adam/batch_normalization_643/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_643/gamma/v*
_output_shapes
:i*
dtype0

#Adam/batch_normalization_643/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*4
shared_name%#Adam/batch_normalization_643/beta/v

7Adam/batch_normalization_643/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_643/beta/v*
_output_shapes
:i*
dtype0

Adam/dense_715/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ii*(
shared_nameAdam/dense_715/kernel/v

+Adam/dense_715/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_715/kernel/v*
_output_shapes

:ii*
dtype0

Adam/dense_715/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*&
shared_nameAdam/dense_715/bias/v
{
)Adam/dense_715/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_715/bias/v*
_output_shapes
:i*
dtype0
 
$Adam/batch_normalization_644/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*5
shared_name&$Adam/batch_normalization_644/gamma/v

8Adam/batch_normalization_644/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_644/gamma/v*
_output_shapes
:i*
dtype0

#Adam/batch_normalization_644/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:i*4
shared_name%#Adam/batch_normalization_644/beta/v

7Adam/batch_normalization_644/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_644/beta/v*
_output_shapes
:i*
dtype0

Adam/dense_716/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:i=*(
shared_nameAdam/dense_716/kernel/v

+Adam/dense_716/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_716/kernel/v*
_output_shapes

:i=*
dtype0

Adam/dense_716/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*&
shared_nameAdam/dense_716/bias/v
{
)Adam/dense_716/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_716/bias/v*
_output_shapes
:=*
dtype0
 
$Adam/batch_normalization_645/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*5
shared_name&$Adam/batch_normalization_645/gamma/v

8Adam/batch_normalization_645/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_645/gamma/v*
_output_shapes
:=*
dtype0

#Adam/batch_normalization_645/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*4
shared_name%#Adam/batch_normalization_645/beta/v

7Adam/batch_normalization_645/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_645/beta/v*
_output_shapes
:=*
dtype0

Adam/dense_717/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:==*(
shared_nameAdam/dense_717/kernel/v

+Adam/dense_717/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_717/kernel/v*
_output_shapes

:==*
dtype0

Adam/dense_717/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*&
shared_nameAdam/dense_717/bias/v
{
)Adam/dense_717/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_717/bias/v*
_output_shapes
:=*
dtype0
 
$Adam/batch_normalization_646/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*5
shared_name&$Adam/batch_normalization_646/gamma/v

8Adam/batch_normalization_646/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_646/gamma/v*
_output_shapes
:=*
dtype0

#Adam/batch_normalization_646/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:=*4
shared_name%#Adam/batch_normalization_646/beta/v

7Adam/batch_normalization_646/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_646/beta/v*
_output_shapes
:=*
dtype0

Adam/dense_718/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:=7*(
shared_nameAdam/dense_718/kernel/v

+Adam/dense_718/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_718/kernel/v*
_output_shapes

:=7*
dtype0

Adam/dense_718/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*&
shared_nameAdam/dense_718/bias/v
{
)Adam/dense_718/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_718/bias/v*
_output_shapes
:7*
dtype0
 
$Adam/batch_normalization_647/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*5
shared_name&$Adam/batch_normalization_647/gamma/v

8Adam/batch_normalization_647/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_647/gamma/v*
_output_shapes
:7*
dtype0

#Adam/batch_normalization_647/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*4
shared_name%#Adam/batch_normalization_647/beta/v

7Adam/batch_normalization_647/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_647/beta/v*
_output_shapes
:7*
dtype0

Adam/dense_719/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:77*(
shared_nameAdam/dense_719/kernel/v

+Adam/dense_719/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_719/kernel/v*
_output_shapes

:77*
dtype0

Adam/dense_719/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*&
shared_nameAdam/dense_719/bias/v
{
)Adam/dense_719/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_719/bias/v*
_output_shapes
:7*
dtype0
 
$Adam/batch_normalization_648/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*5
shared_name&$Adam/batch_normalization_648/gamma/v

8Adam/batch_normalization_648/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_648/gamma/v*
_output_shapes
:7*
dtype0

#Adam/batch_normalization_648/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*4
shared_name%#Adam/batch_normalization_648/beta/v

7Adam/batch_normalization_648/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_648/beta/v*
_output_shapes
:7*
dtype0

Adam/dense_720/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7*(
shared_nameAdam/dense_720/kernel/v

+Adam/dense_720/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_720/kernel/v*
_output_shapes

:7*
dtype0

Adam/dense_720/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_720/bias/v
{
)Adam/dense_720/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_720/bias/v*
_output_shapes
:*
dtype0
f
ConstConst*
_output_shapes

:*
dtype0*)
value B"UUéBÿÿ8B  DA  DA
h
Const_1Const*
_output_shapes

:*
dtype0*)
value B"4sE æD ÀBÿ¿B

NoOpNoOp
¡
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*É 
value¾ Bº  B² 
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

decay0m½1m¾9m¿:mÀImÁJmÂRmÃSmÄbmÅcmÆkmÇlmÈ{mÉ|mÊ	mË	mÌ	mÍ	mÎ	mÏ	mÐ	­mÑ	®mÒ	¶mÓ	·mÔ	ÆmÕ	ÇmÖ	Ïm×	ÐmØ	ßmÙ	àmÚ	èmÛ	émÜ	ømÝ	ùmÞ	mß	mà	má	mâ0vã1vä9vå:væIvçJvèRvéSvêbvëcvìkvílvî{vï|vð	vñ	vò	vó	vô	võ	vö	­v÷	®vø	¶vù	·vú	Ævû	Çvü	Ïvý	Ðvþ	ßvÿ	àv	èv	év	øv	ùv	v	v	v	v*
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
J
0
1
2
 3
¡4
¢5
£6
¤7
¥8* 
µ
¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
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
«serving_default* 
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
VARIABLE_VALUEdense_711/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_711/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

00
11*

00
11*


0* 

¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
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
VARIABLE_VALUEbatch_normalization_640/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_640/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_640/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_640/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
90
:1
;2
<3*

90
:1*
* 

±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
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
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_712/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_712/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

I0
J1*

I0
J1*


0* 

»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
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
VARIABLE_VALUEbatch_normalization_641/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_641/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_641/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_641/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
R0
S1
T2
U3*

R0
S1*
* 

Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
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
Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_713/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_713/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

b0
c1*

b0
c1*


0* 

Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
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
VARIABLE_VALUEbatch_normalization_642/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_642/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_642/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_642/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
k0
l1
m2
n3*

k0
l1*
* 

Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
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
Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_714/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_714/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

{0
|1*

{0
|1*


 0* 

Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
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
VARIABLE_VALUEbatch_normalization_643/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_643/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_643/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_643/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
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
ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_715/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_715/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*


¡0* 

ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
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
VARIABLE_VALUEbatch_normalization_644/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_644/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_644/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_644/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
ínon_trainable_variables
îlayers
ïmetrics
 ðlayer_regularization_losses
ñlayer_metrics
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
ònon_trainable_variables
ólayers
ômetrics
 õlayer_regularization_losses
ölayer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_716/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_716/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

­0
®1*

­0
®1*


¢0* 

÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
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
VARIABLE_VALUEbatch_normalization_645/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_645/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_645/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_645/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
À	variables
Átrainable_variables
Âregularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_717/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_717/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

Æ0
Ç1*

Æ0
Ç1*


£0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
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
VARIABLE_VALUEbatch_normalization_646/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_646/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_646/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_646/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ù	variables
Útrainable_variables
Ûregularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_718/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_718/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

ß0
à1*

ß0
à1*


¤0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
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
VARIABLE_VALUEbatch_normalization_647/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_647/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_647/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_647/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
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
non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
ò	variables
ótrainable_variables
ôregularization_losses
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_719/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_719/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*

ø0
ù1*

ø0
ù1*


¥0* 

¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
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
VARIABLE_VALUEbatch_normalization_648/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_648/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_648/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_648/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
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
®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_720/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_720/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
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
* 
* 
* 
* 
* 
* 
* 
* 
* 
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

¸0*
* 
* 
* 
* 
* 
* 


0* 
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


0* 
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


0* 
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


 0* 
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


¡0* 
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


¢0* 
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


£0* 
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


¤0* 
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


¥0* 
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

¹total

ºcount
»	variables
¼	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

¹0
º1*

»	variables*
}
VARIABLE_VALUEAdam/dense_711/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_711/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_640/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_640/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_712/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_712/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_641/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_641/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_713/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_713/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_642/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_642/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_714/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_714/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_643/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_643/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_715/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_715/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_644/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_644/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_716/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_716/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_645/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_645/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_717/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_717/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_646/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_646/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_718/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_718/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_647/gamma/mRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_647/beta/mQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_719/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_719/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_648/gamma/mRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_648/beta/mQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_720/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_720/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_711/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_711/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_640/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_640/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_712/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_712/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_641/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_641/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_713/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_713/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_642/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_642/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_714/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_714/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_643/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_643/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_715/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_715/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_644/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_644/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_716/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_716/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_645/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_645/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_717/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_717/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_646/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_646/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_718/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_718/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_647/gamma/vRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_647/beta/vQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_719/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_719/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_648/gamma/vRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_648/beta/vQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_720/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_720/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

&serving_default_normalization_71_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
û
StatefulPartitionedCallStatefulPartitionedCall&serving_default_normalization_71_inputConstConst_1dense_711/kerneldense_711/bias'batch_normalization_640/moving_variancebatch_normalization_640/gamma#batch_normalization_640/moving_meanbatch_normalization_640/betadense_712/kerneldense_712/bias'batch_normalization_641/moving_variancebatch_normalization_641/gamma#batch_normalization_641/moving_meanbatch_normalization_641/betadense_713/kerneldense_713/bias'batch_normalization_642/moving_variancebatch_normalization_642/gamma#batch_normalization_642/moving_meanbatch_normalization_642/betadense_714/kerneldense_714/bias'batch_normalization_643/moving_variancebatch_normalization_643/gamma#batch_normalization_643/moving_meanbatch_normalization_643/betadense_715/kerneldense_715/bias'batch_normalization_644/moving_variancebatch_normalization_644/gamma#batch_normalization_644/moving_meanbatch_normalization_644/betadense_716/kerneldense_716/bias'batch_normalization_645/moving_variancebatch_normalization_645/gamma#batch_normalization_645/moving_meanbatch_normalization_645/betadense_717/kerneldense_717/bias'batch_normalization_646/moving_variancebatch_normalization_646/gamma#batch_normalization_646/moving_meanbatch_normalization_646/betadense_718/kerneldense_718/bias'batch_normalization_647/moving_variancebatch_normalization_647/gamma#batch_normalization_647/moving_meanbatch_normalization_647/betadense_719/kerneldense_719/bias'batch_normalization_648/moving_variancebatch_normalization_648/gamma#batch_normalization_648/moving_meanbatch_normalization_648/betadense_720/kerneldense_720/bias*F
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
GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1124930
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
È8
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp$dense_711/kernel/Read/ReadVariableOp"dense_711/bias/Read/ReadVariableOp1batch_normalization_640/gamma/Read/ReadVariableOp0batch_normalization_640/beta/Read/ReadVariableOp7batch_normalization_640/moving_mean/Read/ReadVariableOp;batch_normalization_640/moving_variance/Read/ReadVariableOp$dense_712/kernel/Read/ReadVariableOp"dense_712/bias/Read/ReadVariableOp1batch_normalization_641/gamma/Read/ReadVariableOp0batch_normalization_641/beta/Read/ReadVariableOp7batch_normalization_641/moving_mean/Read/ReadVariableOp;batch_normalization_641/moving_variance/Read/ReadVariableOp$dense_713/kernel/Read/ReadVariableOp"dense_713/bias/Read/ReadVariableOp1batch_normalization_642/gamma/Read/ReadVariableOp0batch_normalization_642/beta/Read/ReadVariableOp7batch_normalization_642/moving_mean/Read/ReadVariableOp;batch_normalization_642/moving_variance/Read/ReadVariableOp$dense_714/kernel/Read/ReadVariableOp"dense_714/bias/Read/ReadVariableOp1batch_normalization_643/gamma/Read/ReadVariableOp0batch_normalization_643/beta/Read/ReadVariableOp7batch_normalization_643/moving_mean/Read/ReadVariableOp;batch_normalization_643/moving_variance/Read/ReadVariableOp$dense_715/kernel/Read/ReadVariableOp"dense_715/bias/Read/ReadVariableOp1batch_normalization_644/gamma/Read/ReadVariableOp0batch_normalization_644/beta/Read/ReadVariableOp7batch_normalization_644/moving_mean/Read/ReadVariableOp;batch_normalization_644/moving_variance/Read/ReadVariableOp$dense_716/kernel/Read/ReadVariableOp"dense_716/bias/Read/ReadVariableOp1batch_normalization_645/gamma/Read/ReadVariableOp0batch_normalization_645/beta/Read/ReadVariableOp7batch_normalization_645/moving_mean/Read/ReadVariableOp;batch_normalization_645/moving_variance/Read/ReadVariableOp$dense_717/kernel/Read/ReadVariableOp"dense_717/bias/Read/ReadVariableOp1batch_normalization_646/gamma/Read/ReadVariableOp0batch_normalization_646/beta/Read/ReadVariableOp7batch_normalization_646/moving_mean/Read/ReadVariableOp;batch_normalization_646/moving_variance/Read/ReadVariableOp$dense_718/kernel/Read/ReadVariableOp"dense_718/bias/Read/ReadVariableOp1batch_normalization_647/gamma/Read/ReadVariableOp0batch_normalization_647/beta/Read/ReadVariableOp7batch_normalization_647/moving_mean/Read/ReadVariableOp;batch_normalization_647/moving_variance/Read/ReadVariableOp$dense_719/kernel/Read/ReadVariableOp"dense_719/bias/Read/ReadVariableOp1batch_normalization_648/gamma/Read/ReadVariableOp0batch_normalization_648/beta/Read/ReadVariableOp7batch_normalization_648/moving_mean/Read/ReadVariableOp;batch_normalization_648/moving_variance/Read/ReadVariableOp$dense_720/kernel/Read/ReadVariableOp"dense_720/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_711/kernel/m/Read/ReadVariableOp)Adam/dense_711/bias/m/Read/ReadVariableOp8Adam/batch_normalization_640/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_640/beta/m/Read/ReadVariableOp+Adam/dense_712/kernel/m/Read/ReadVariableOp)Adam/dense_712/bias/m/Read/ReadVariableOp8Adam/batch_normalization_641/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_641/beta/m/Read/ReadVariableOp+Adam/dense_713/kernel/m/Read/ReadVariableOp)Adam/dense_713/bias/m/Read/ReadVariableOp8Adam/batch_normalization_642/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_642/beta/m/Read/ReadVariableOp+Adam/dense_714/kernel/m/Read/ReadVariableOp)Adam/dense_714/bias/m/Read/ReadVariableOp8Adam/batch_normalization_643/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_643/beta/m/Read/ReadVariableOp+Adam/dense_715/kernel/m/Read/ReadVariableOp)Adam/dense_715/bias/m/Read/ReadVariableOp8Adam/batch_normalization_644/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_644/beta/m/Read/ReadVariableOp+Adam/dense_716/kernel/m/Read/ReadVariableOp)Adam/dense_716/bias/m/Read/ReadVariableOp8Adam/batch_normalization_645/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_645/beta/m/Read/ReadVariableOp+Adam/dense_717/kernel/m/Read/ReadVariableOp)Adam/dense_717/bias/m/Read/ReadVariableOp8Adam/batch_normalization_646/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_646/beta/m/Read/ReadVariableOp+Adam/dense_718/kernel/m/Read/ReadVariableOp)Adam/dense_718/bias/m/Read/ReadVariableOp8Adam/batch_normalization_647/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_647/beta/m/Read/ReadVariableOp+Adam/dense_719/kernel/m/Read/ReadVariableOp)Adam/dense_719/bias/m/Read/ReadVariableOp8Adam/batch_normalization_648/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_648/beta/m/Read/ReadVariableOp+Adam/dense_720/kernel/m/Read/ReadVariableOp)Adam/dense_720/bias/m/Read/ReadVariableOp+Adam/dense_711/kernel/v/Read/ReadVariableOp)Adam/dense_711/bias/v/Read/ReadVariableOp8Adam/batch_normalization_640/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_640/beta/v/Read/ReadVariableOp+Adam/dense_712/kernel/v/Read/ReadVariableOp)Adam/dense_712/bias/v/Read/ReadVariableOp8Adam/batch_normalization_641/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_641/beta/v/Read/ReadVariableOp+Adam/dense_713/kernel/v/Read/ReadVariableOp)Adam/dense_713/bias/v/Read/ReadVariableOp8Adam/batch_normalization_642/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_642/beta/v/Read/ReadVariableOp+Adam/dense_714/kernel/v/Read/ReadVariableOp)Adam/dense_714/bias/v/Read/ReadVariableOp8Adam/batch_normalization_643/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_643/beta/v/Read/ReadVariableOp+Adam/dense_715/kernel/v/Read/ReadVariableOp)Adam/dense_715/bias/v/Read/ReadVariableOp8Adam/batch_normalization_644/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_644/beta/v/Read/ReadVariableOp+Adam/dense_716/kernel/v/Read/ReadVariableOp)Adam/dense_716/bias/v/Read/ReadVariableOp8Adam/batch_normalization_645/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_645/beta/v/Read/ReadVariableOp+Adam/dense_717/kernel/v/Read/ReadVariableOp)Adam/dense_717/bias/v/Read/ReadVariableOp8Adam/batch_normalization_646/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_646/beta/v/Read/ReadVariableOp+Adam/dense_718/kernel/v/Read/ReadVariableOp)Adam/dense_718/bias/v/Read/ReadVariableOp8Adam/batch_normalization_647/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_647/beta/v/Read/ReadVariableOp+Adam/dense_719/kernel/v/Read/ReadVariableOp)Adam/dense_719/bias/v/Read/ReadVariableOp8Adam/batch_normalization_648/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_648/beta/v/Read/ReadVariableOp+Adam/dense_720/kernel/v/Read/ReadVariableOp)Adam/dense_720/bias/v/Read/ReadVariableOpConst_2*
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_1126875
½"
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_711/kerneldense_711/biasbatch_normalization_640/gammabatch_normalization_640/beta#batch_normalization_640/moving_mean'batch_normalization_640/moving_variancedense_712/kerneldense_712/biasbatch_normalization_641/gammabatch_normalization_641/beta#batch_normalization_641/moving_mean'batch_normalization_641/moving_variancedense_713/kerneldense_713/biasbatch_normalization_642/gammabatch_normalization_642/beta#batch_normalization_642/moving_mean'batch_normalization_642/moving_variancedense_714/kerneldense_714/biasbatch_normalization_643/gammabatch_normalization_643/beta#batch_normalization_643/moving_mean'batch_normalization_643/moving_variancedense_715/kerneldense_715/biasbatch_normalization_644/gammabatch_normalization_644/beta#batch_normalization_644/moving_mean'batch_normalization_644/moving_variancedense_716/kerneldense_716/biasbatch_normalization_645/gammabatch_normalization_645/beta#batch_normalization_645/moving_mean'batch_normalization_645/moving_variancedense_717/kerneldense_717/biasbatch_normalization_646/gammabatch_normalization_646/beta#batch_normalization_646/moving_mean'batch_normalization_646/moving_variancedense_718/kerneldense_718/biasbatch_normalization_647/gammabatch_normalization_647/beta#batch_normalization_647/moving_mean'batch_normalization_647/moving_variancedense_719/kerneldense_719/biasbatch_normalization_648/gammabatch_normalization_648/beta#batch_normalization_648/moving_mean'batch_normalization_648/moving_variancedense_720/kerneldense_720/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount_1Adam/dense_711/kernel/mAdam/dense_711/bias/m$Adam/batch_normalization_640/gamma/m#Adam/batch_normalization_640/beta/mAdam/dense_712/kernel/mAdam/dense_712/bias/m$Adam/batch_normalization_641/gamma/m#Adam/batch_normalization_641/beta/mAdam/dense_713/kernel/mAdam/dense_713/bias/m$Adam/batch_normalization_642/gamma/m#Adam/batch_normalization_642/beta/mAdam/dense_714/kernel/mAdam/dense_714/bias/m$Adam/batch_normalization_643/gamma/m#Adam/batch_normalization_643/beta/mAdam/dense_715/kernel/mAdam/dense_715/bias/m$Adam/batch_normalization_644/gamma/m#Adam/batch_normalization_644/beta/mAdam/dense_716/kernel/mAdam/dense_716/bias/m$Adam/batch_normalization_645/gamma/m#Adam/batch_normalization_645/beta/mAdam/dense_717/kernel/mAdam/dense_717/bias/m$Adam/batch_normalization_646/gamma/m#Adam/batch_normalization_646/beta/mAdam/dense_718/kernel/mAdam/dense_718/bias/m$Adam/batch_normalization_647/gamma/m#Adam/batch_normalization_647/beta/mAdam/dense_719/kernel/mAdam/dense_719/bias/m$Adam/batch_normalization_648/gamma/m#Adam/batch_normalization_648/beta/mAdam/dense_720/kernel/mAdam/dense_720/bias/mAdam/dense_711/kernel/vAdam/dense_711/bias/v$Adam/batch_normalization_640/gamma/v#Adam/batch_normalization_640/beta/vAdam/dense_712/kernel/vAdam/dense_712/bias/v$Adam/batch_normalization_641/gamma/v#Adam/batch_normalization_641/beta/vAdam/dense_713/kernel/vAdam/dense_713/bias/v$Adam/batch_normalization_642/gamma/v#Adam/batch_normalization_642/beta/vAdam/dense_714/kernel/vAdam/dense_714/bias/v$Adam/batch_normalization_643/gamma/v#Adam/batch_normalization_643/beta/vAdam/dense_715/kernel/vAdam/dense_715/bias/v$Adam/batch_normalization_644/gamma/v#Adam/batch_normalization_644/beta/vAdam/dense_716/kernel/vAdam/dense_716/bias/v$Adam/batch_normalization_645/gamma/v#Adam/batch_normalization_645/beta/vAdam/dense_717/kernel/vAdam/dense_717/bias/v$Adam/batch_normalization_646/gamma/v#Adam/batch_normalization_646/beta/vAdam/dense_718/kernel/vAdam/dense_718/bias/v$Adam/batch_normalization_647/gamma/v#Adam/batch_normalization_647/beta/vAdam/dense_719/kernel/vAdam/dense_719/bias/v$Adam/batch_normalization_648/gamma/v#Adam/batch_normalization_648/beta/vAdam/dense_720/kernel/vAdam/dense_720/bias/v*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_1127308ïÅ/
¬
Ô
9__inference_batch_normalization_643_layer_call_fn_1125469

inputs
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_643_layer_call_and_return_conditional_losses_1121078o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_641_layer_call_fn_1125191

inputs
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_641_layer_call_and_return_conditional_losses_1120914o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_646_layer_call_and_return_conditional_losses_1125906

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
æ
h
L__inference_leaky_re_lu_648_layer_call_and_return_conditional_losses_1126228

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
Æ

+__inference_dense_718_layer_call_fn_1125974

inputs
unknown:=7
	unknown_0:7
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_718_layer_call_and_return_conditional_losses_1121867o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7`
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
Ç
Ó
/__inference_sequential_71_layer_call_fn_1123963

inputs
unknown
	unknown_0
	unknown_1:i
	unknown_2:i
	unknown_3:i
	unknown_4:i
	unknown_5:i
	unknown_6:i
	unknown_7:ii
	unknown_8:i
	unknown_9:i

unknown_10:i

unknown_11:i

unknown_12:i

unknown_13:ii

unknown_14:i

unknown_15:i

unknown_16:i

unknown_17:i

unknown_18:i

unknown_19:ii

unknown_20:i

unknown_21:i

unknown_22:i

unknown_23:i

unknown_24:i

unknown_25:ii

unknown_26:i

unknown_27:i

unknown_28:i

unknown_29:i

unknown_30:i

unknown_31:i=

unknown_32:=

unknown_33:=

unknown_34:=

unknown_35:=

unknown_36:=

unknown_37:==

unknown_38:=

unknown_39:=

unknown_40:=

unknown_41:=

unknown_42:=

unknown_43:=7

unknown_44:7

unknown_45:7

unknown_46:7

unknown_47:7

unknown_48:7

unknown_49:77

unknown_50:7

unknown_51:7

unknown_52:7

unknown_53:7

unknown_54:7

unknown_55:7

unknown_56:
identity¢StatefulPartitionedCallÒ
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
GPU 2J 8 *S
fNRL
J__inference_sequential_71_layer_call_and_return_conditional_losses_1122770o
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
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Æ

+__inference_dense_712_layer_call_fn_1125140

inputs
unknown:ii
	unknown_0:i
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_712_layer_call_and_return_conditional_losses_1121585o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_642_layer_call_fn_1125317

inputs
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_642_layer_call_and_return_conditional_losses_1120949o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_642_layer_call_fn_1125330

inputs
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_642_layer_call_and_return_conditional_losses_1120996o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_643_layer_call_and_return_conditional_losses_1121031

inputs/
!batchnorm_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i1
#batchnorm_readvariableop_1_resource:i1
#batchnorm_readvariableop_2_resource:i
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:iz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
¥
Þ
F__inference_dense_715_layer_call_and_return_conditional_losses_1125582

inputs0
matmul_readvariableop_resource:ii-
biasadd_readvariableop_resource:i
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_715/kernel/Regularizer/Abs/ReadVariableOp¢2dense_715/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿir
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:i*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿig
"dense_715/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_715/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_715/kernel/Regularizer/AbsAbs7dense_715/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_715/kernel/Regularizer/SumSum$dense_715/kernel/Regularizer/Abs:y:0-dense_715/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_715/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_715/kernel/Regularizer/mulMul+dense_715/kernel/Regularizer/mul/x:output:0)dense_715/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_715/kernel/Regularizer/addAddV2+dense_715/kernel/Regularizer/Const:output:0$dense_715/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_715/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_715/kernel/Regularizer/SquareSquare:dense_715/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_715/kernel/Regularizer/Sum_1Sum'dense_715/kernel/Regularizer/Square:y:0-dense_715/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_715/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_715/kernel/Regularizer/mul_1Mul-dense_715/kernel/Regularizer/mul_1/x:output:0+dense_715/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_715/kernel/Regularizer/add_1AddV2$dense_715/kernel/Regularizer/add:z:0&dense_715/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÞ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_715/kernel/Regularizer/Abs/ReadVariableOp3^dense_715/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_715/kernel/Regularizer/Abs/ReadVariableOp/dense_715/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_715/kernel/Regularizer/Square/ReadVariableOp2dense_715/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
Æ

+__inference_dense_711_layer_call_fn_1125001

inputs
unknown:i
	unknown_0:i
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_711_layer_call_and_return_conditional_losses_1121538o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi`
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
¥
Þ
F__inference_dense_714_layer_call_and_return_conditional_losses_1125443

inputs0
matmul_readvariableop_resource:ii-
biasadd_readvariableop_resource:i
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_714/kernel/Regularizer/Abs/ReadVariableOp¢2dense_714/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿir
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:i*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿig
"dense_714/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_714/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_714/kernel/Regularizer/AbsAbs7dense_714/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_714/kernel/Regularizer/SumSum$dense_714/kernel/Regularizer/Abs:y:0-dense_714/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_714/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_714/kernel/Regularizer/mulMul+dense_714/kernel/Regularizer/mul/x:output:0)dense_714/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_714/kernel/Regularizer/addAddV2+dense_714/kernel/Regularizer/Const:output:0$dense_714/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_714/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_714/kernel/Regularizer/SquareSquare:dense_714/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_714/kernel/Regularizer/Sum_1Sum'dense_714/kernel/Regularizer/Square:y:0-dense_714/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_714/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_714/kernel/Regularizer/mul_1Mul-dense_714/kernel/Regularizer/mul_1/x:output:0+dense_714/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_714/kernel/Regularizer/add_1AddV2$dense_714/kernel/Regularizer/add:z:0&dense_714/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÞ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_714/kernel/Regularizer/Abs/ReadVariableOp3^dense_714/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_714/kernel/Regularizer/Abs/ReadVariableOp/dense_714/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_714/kernel/Regularizer/Square/ReadVariableOp2dense_714/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_643_layer_call_and_return_conditional_losses_1125533

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿi:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs

ã
__inference_loss_fn_1_1126287J
8dense_712_kernel_regularizer_abs_readvariableop_resource:ii
identity¢/dense_712/kernel/Regularizer/Abs/ReadVariableOp¢2dense_712/kernel/Regularizer/Square/ReadVariableOpg
"dense_712/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
/dense_712/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8dense_712_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_712/kernel/Regularizer/AbsAbs7dense_712/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_712/kernel/Regularizer/SumSum$dense_712/kernel/Regularizer/Abs:y:0-dense_712/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_712/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_712/kernel/Regularizer/mulMul+dense_712/kernel/Regularizer/mul/x:output:0)dense_712/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_712/kernel/Regularizer/addAddV2+dense_712/kernel/Regularizer/Const:output:0$dense_712/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: «
2dense_712/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense_712_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_712/kernel/Regularizer/SquareSquare:dense_712/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_712/kernel/Regularizer/Sum_1Sum'dense_712/kernel/Regularizer/Square:y:0-dense_712/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_712/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_712/kernel/Regularizer/mul_1Mul-dense_712/kernel/Regularizer/mul_1/x:output:0+dense_712/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_712/kernel/Regularizer/add_1AddV2$dense_712/kernel/Regularizer/add:z:0&dense_712/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
IdentityIdentity&dense_712/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ­
NoOpNoOp0^dense_712/kernel/Regularizer/Abs/ReadVariableOp3^dense_712/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_712/kernel/Regularizer/Abs/ReadVariableOp/dense_712/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_712/kernel/Regularizer/Square/ReadVariableOp2dense_712/kernel/Regularizer/Square/ReadVariableOp
¥
Þ
F__inference_dense_719_layer_call_and_return_conditional_losses_1121914

inputs0
matmul_readvariableop_resource:77-
biasadd_readvariableop_resource:7
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_719/kernel/Regularizer/Abs/ReadVariableOp¢2dense_719/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:77*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:7*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7g
"dense_719/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_719/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:77*
dtype0
 dense_719/kernel/Regularizer/AbsAbs7dense_719/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_719/kernel/Regularizer/SumSum$dense_719/kernel/Regularizer/Abs:y:0-dense_719/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_719/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_719/kernel/Regularizer/mulMul+dense_719/kernel/Regularizer/mul/x:output:0)dense_719/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_719/kernel/Regularizer/addAddV2+dense_719/kernel/Regularizer/Const:output:0$dense_719/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_719/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:77*
dtype0
#dense_719/kernel/Regularizer/SquareSquare:dense_719/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_719/kernel/Regularizer/Sum_1Sum'dense_719/kernel/Regularizer/Square:y:0-dense_719/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_719/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_719/kernel/Regularizer/mul_1Mul-dense_719/kernel/Regularizer/mul_1/x:output:0+dense_719/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_719/kernel/Regularizer/add_1AddV2$dense_719/kernel/Regularizer/add:z:0&dense_719/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7Þ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_719/kernel/Regularizer/Abs/ReadVariableOp3^dense_719/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_719/kernel/Regularizer/Abs/ReadVariableOp/dense_719/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_719/kernel/Regularizer/Square/ReadVariableOp2dense_719/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
«Ç
Ó!
J__inference_sequential_71_layer_call_and_return_conditional_losses_1123296
normalization_71_input
normalization_71_sub_y
normalization_71_sqrt_x#
dense_711_1123020:i
dense_711_1123022:i-
batch_normalization_640_1123025:i-
batch_normalization_640_1123027:i-
batch_normalization_640_1123029:i-
batch_normalization_640_1123031:i#
dense_712_1123035:ii
dense_712_1123037:i-
batch_normalization_641_1123040:i-
batch_normalization_641_1123042:i-
batch_normalization_641_1123044:i-
batch_normalization_641_1123046:i#
dense_713_1123050:ii
dense_713_1123052:i-
batch_normalization_642_1123055:i-
batch_normalization_642_1123057:i-
batch_normalization_642_1123059:i-
batch_normalization_642_1123061:i#
dense_714_1123065:ii
dense_714_1123067:i-
batch_normalization_643_1123070:i-
batch_normalization_643_1123072:i-
batch_normalization_643_1123074:i-
batch_normalization_643_1123076:i#
dense_715_1123080:ii
dense_715_1123082:i-
batch_normalization_644_1123085:i-
batch_normalization_644_1123087:i-
batch_normalization_644_1123089:i-
batch_normalization_644_1123091:i#
dense_716_1123095:i=
dense_716_1123097:=-
batch_normalization_645_1123100:=-
batch_normalization_645_1123102:=-
batch_normalization_645_1123104:=-
batch_normalization_645_1123106:=#
dense_717_1123110:==
dense_717_1123112:=-
batch_normalization_646_1123115:=-
batch_normalization_646_1123117:=-
batch_normalization_646_1123119:=-
batch_normalization_646_1123121:=#
dense_718_1123125:=7
dense_718_1123127:7-
batch_normalization_647_1123130:7-
batch_normalization_647_1123132:7-
batch_normalization_647_1123134:7-
batch_normalization_647_1123136:7#
dense_719_1123140:77
dense_719_1123142:7-
batch_normalization_648_1123145:7-
batch_normalization_648_1123147:7-
batch_normalization_648_1123149:7-
batch_normalization_648_1123151:7#
dense_720_1123155:7
dense_720_1123157:
identity¢/batch_normalization_640/StatefulPartitionedCall¢/batch_normalization_641/StatefulPartitionedCall¢/batch_normalization_642/StatefulPartitionedCall¢/batch_normalization_643/StatefulPartitionedCall¢/batch_normalization_644/StatefulPartitionedCall¢/batch_normalization_645/StatefulPartitionedCall¢/batch_normalization_646/StatefulPartitionedCall¢/batch_normalization_647/StatefulPartitionedCall¢/batch_normalization_648/StatefulPartitionedCall¢!dense_711/StatefulPartitionedCall¢/dense_711/kernel/Regularizer/Abs/ReadVariableOp¢2dense_711/kernel/Regularizer/Square/ReadVariableOp¢!dense_712/StatefulPartitionedCall¢/dense_712/kernel/Regularizer/Abs/ReadVariableOp¢2dense_712/kernel/Regularizer/Square/ReadVariableOp¢!dense_713/StatefulPartitionedCall¢/dense_713/kernel/Regularizer/Abs/ReadVariableOp¢2dense_713/kernel/Regularizer/Square/ReadVariableOp¢!dense_714/StatefulPartitionedCall¢/dense_714/kernel/Regularizer/Abs/ReadVariableOp¢2dense_714/kernel/Regularizer/Square/ReadVariableOp¢!dense_715/StatefulPartitionedCall¢/dense_715/kernel/Regularizer/Abs/ReadVariableOp¢2dense_715/kernel/Regularizer/Square/ReadVariableOp¢!dense_716/StatefulPartitionedCall¢/dense_716/kernel/Regularizer/Abs/ReadVariableOp¢2dense_716/kernel/Regularizer/Square/ReadVariableOp¢!dense_717/StatefulPartitionedCall¢/dense_717/kernel/Regularizer/Abs/ReadVariableOp¢2dense_717/kernel/Regularizer/Square/ReadVariableOp¢!dense_718/StatefulPartitionedCall¢/dense_718/kernel/Regularizer/Abs/ReadVariableOp¢2dense_718/kernel/Regularizer/Square/ReadVariableOp¢!dense_719/StatefulPartitionedCall¢/dense_719/kernel/Regularizer/Abs/ReadVariableOp¢2dense_719/kernel/Regularizer/Square/ReadVariableOp¢!dense_720/StatefulPartitionedCall}
normalization_71/subSubnormalization_71_inputnormalization_71_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_71/SqrtSqrtnormalization_71_sqrt_x*
T0*
_output_shapes

:_
normalization_71/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_71/MaximumMaximumnormalization_71/Sqrt:y:0#normalization_71/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_71/truedivRealDivnormalization_71/sub:z:0normalization_71/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_711/StatefulPartitionedCallStatefulPartitionedCallnormalization_71/truediv:z:0dense_711_1123020dense_711_1123022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_711_layer_call_and_return_conditional_losses_1121538
/batch_normalization_640/StatefulPartitionedCallStatefulPartitionedCall*dense_711/StatefulPartitionedCall:output:0batch_normalization_640_1123025batch_normalization_640_1123027batch_normalization_640_1123029batch_normalization_640_1123031*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_640_layer_call_and_return_conditional_losses_1120785ù
leaky_re_lu_640/PartitionedCallPartitionedCall8batch_normalization_640/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_640_layer_call_and_return_conditional_losses_1121558
!dense_712/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_640/PartitionedCall:output:0dense_712_1123035dense_712_1123037*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_712_layer_call_and_return_conditional_losses_1121585
/batch_normalization_641/StatefulPartitionedCallStatefulPartitionedCall*dense_712/StatefulPartitionedCall:output:0batch_normalization_641_1123040batch_normalization_641_1123042batch_normalization_641_1123044batch_normalization_641_1123046*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_641_layer_call_and_return_conditional_losses_1120867ù
leaky_re_lu_641/PartitionedCallPartitionedCall8batch_normalization_641/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_641_layer_call_and_return_conditional_losses_1121605
!dense_713/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_641/PartitionedCall:output:0dense_713_1123050dense_713_1123052*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_713_layer_call_and_return_conditional_losses_1121632
/batch_normalization_642/StatefulPartitionedCallStatefulPartitionedCall*dense_713/StatefulPartitionedCall:output:0batch_normalization_642_1123055batch_normalization_642_1123057batch_normalization_642_1123059batch_normalization_642_1123061*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_642_layer_call_and_return_conditional_losses_1120949ù
leaky_re_lu_642/PartitionedCallPartitionedCall8batch_normalization_642/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_642_layer_call_and_return_conditional_losses_1121652
!dense_714/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_642/PartitionedCall:output:0dense_714_1123065dense_714_1123067*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_714_layer_call_and_return_conditional_losses_1121679
/batch_normalization_643/StatefulPartitionedCallStatefulPartitionedCall*dense_714/StatefulPartitionedCall:output:0batch_normalization_643_1123070batch_normalization_643_1123072batch_normalization_643_1123074batch_normalization_643_1123076*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_643_layer_call_and_return_conditional_losses_1121031ù
leaky_re_lu_643/PartitionedCallPartitionedCall8batch_normalization_643/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_643_layer_call_and_return_conditional_losses_1121699
!dense_715/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_643/PartitionedCall:output:0dense_715_1123080dense_715_1123082*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_715_layer_call_and_return_conditional_losses_1121726
/batch_normalization_644/StatefulPartitionedCallStatefulPartitionedCall*dense_715/StatefulPartitionedCall:output:0batch_normalization_644_1123085batch_normalization_644_1123087batch_normalization_644_1123089batch_normalization_644_1123091*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_644_layer_call_and_return_conditional_losses_1121113ù
leaky_re_lu_644/PartitionedCallPartitionedCall8batch_normalization_644/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_644_layer_call_and_return_conditional_losses_1121746
!dense_716/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_644/PartitionedCall:output:0dense_716_1123095dense_716_1123097*
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
GPU 2J 8 *O
fJRH
F__inference_dense_716_layer_call_and_return_conditional_losses_1121773
/batch_normalization_645/StatefulPartitionedCallStatefulPartitionedCall*dense_716/StatefulPartitionedCall:output:0batch_normalization_645_1123100batch_normalization_645_1123102batch_normalization_645_1123104batch_normalization_645_1123106*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_645_layer_call_and_return_conditional_losses_1121195ù
leaky_re_lu_645/PartitionedCallPartitionedCall8batch_normalization_645/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_645_layer_call_and_return_conditional_losses_1121793
!dense_717/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_645/PartitionedCall:output:0dense_717_1123110dense_717_1123112*
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
GPU 2J 8 *O
fJRH
F__inference_dense_717_layer_call_and_return_conditional_losses_1121820
/batch_normalization_646/StatefulPartitionedCallStatefulPartitionedCall*dense_717/StatefulPartitionedCall:output:0batch_normalization_646_1123115batch_normalization_646_1123117batch_normalization_646_1123119batch_normalization_646_1123121*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_646_layer_call_and_return_conditional_losses_1121277ù
leaky_re_lu_646/PartitionedCallPartitionedCall8batch_normalization_646/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_646_layer_call_and_return_conditional_losses_1121840
!dense_718/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_646/PartitionedCall:output:0dense_718_1123125dense_718_1123127*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_718_layer_call_and_return_conditional_losses_1121867
/batch_normalization_647/StatefulPartitionedCallStatefulPartitionedCall*dense_718/StatefulPartitionedCall:output:0batch_normalization_647_1123130batch_normalization_647_1123132batch_normalization_647_1123134batch_normalization_647_1123136*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_647_layer_call_and_return_conditional_losses_1121359ù
leaky_re_lu_647/PartitionedCallPartitionedCall8batch_normalization_647/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_647_layer_call_and_return_conditional_losses_1121887
!dense_719/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_647/PartitionedCall:output:0dense_719_1123140dense_719_1123142*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_719_layer_call_and_return_conditional_losses_1121914
/batch_normalization_648/StatefulPartitionedCallStatefulPartitionedCall*dense_719/StatefulPartitionedCall:output:0batch_normalization_648_1123145batch_normalization_648_1123147batch_normalization_648_1123149batch_normalization_648_1123151*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_648_layer_call_and_return_conditional_losses_1121441ù
leaky_re_lu_648/PartitionedCallPartitionedCall8batch_normalization_648/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_648_layer_call_and_return_conditional_losses_1121934
!dense_720/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_648/PartitionedCall:output:0dense_720_1123155dense_720_1123157*
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
F__inference_dense_720_layer_call_and_return_conditional_losses_1121946g
"dense_711/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_711/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_711_1123020*
_output_shapes

:i*
dtype0
 dense_711/kernel/Regularizer/AbsAbs7dense_711/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_711/kernel/Regularizer/SumSum$dense_711/kernel/Regularizer/Abs:y:0-dense_711/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_711/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_711/kernel/Regularizer/mulMul+dense_711/kernel/Regularizer/mul/x:output:0)dense_711/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_711/kernel/Regularizer/addAddV2+dense_711/kernel/Regularizer/Const:output:0$dense_711/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_711/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_711_1123020*
_output_shapes

:i*
dtype0
#dense_711/kernel/Regularizer/SquareSquare:dense_711/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_711/kernel/Regularizer/Sum_1Sum'dense_711/kernel/Regularizer/Square:y:0-dense_711/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_711/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_711/kernel/Regularizer/mul_1Mul-dense_711/kernel/Regularizer/mul_1/x:output:0+dense_711/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_711/kernel/Regularizer/add_1AddV2$dense_711/kernel/Regularizer/add:z:0&dense_711/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_712/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_712/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_712_1123035*
_output_shapes

:ii*
dtype0
 dense_712/kernel/Regularizer/AbsAbs7dense_712/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_712/kernel/Regularizer/SumSum$dense_712/kernel/Regularizer/Abs:y:0-dense_712/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_712/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_712/kernel/Regularizer/mulMul+dense_712/kernel/Regularizer/mul/x:output:0)dense_712/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_712/kernel/Regularizer/addAddV2+dense_712/kernel/Regularizer/Const:output:0$dense_712/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_712/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_712_1123035*
_output_shapes

:ii*
dtype0
#dense_712/kernel/Regularizer/SquareSquare:dense_712/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_712/kernel/Regularizer/Sum_1Sum'dense_712/kernel/Regularizer/Square:y:0-dense_712/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_712/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_712/kernel/Regularizer/mul_1Mul-dense_712/kernel/Regularizer/mul_1/x:output:0+dense_712/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_712/kernel/Regularizer/add_1AddV2$dense_712/kernel/Regularizer/add:z:0&dense_712/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_713/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_713/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_713_1123050*
_output_shapes

:ii*
dtype0
 dense_713/kernel/Regularizer/AbsAbs7dense_713/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_713/kernel/Regularizer/SumSum$dense_713/kernel/Regularizer/Abs:y:0-dense_713/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_713/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_713/kernel/Regularizer/mulMul+dense_713/kernel/Regularizer/mul/x:output:0)dense_713/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_713/kernel/Regularizer/addAddV2+dense_713/kernel/Regularizer/Const:output:0$dense_713/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_713/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_713_1123050*
_output_shapes

:ii*
dtype0
#dense_713/kernel/Regularizer/SquareSquare:dense_713/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_713/kernel/Regularizer/Sum_1Sum'dense_713/kernel/Regularizer/Square:y:0-dense_713/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_713/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_713/kernel/Regularizer/mul_1Mul-dense_713/kernel/Regularizer/mul_1/x:output:0+dense_713/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_713/kernel/Regularizer/add_1AddV2$dense_713/kernel/Regularizer/add:z:0&dense_713/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_714/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_714/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_714_1123065*
_output_shapes

:ii*
dtype0
 dense_714/kernel/Regularizer/AbsAbs7dense_714/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_714/kernel/Regularizer/SumSum$dense_714/kernel/Regularizer/Abs:y:0-dense_714/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_714/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_714/kernel/Regularizer/mulMul+dense_714/kernel/Regularizer/mul/x:output:0)dense_714/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_714/kernel/Regularizer/addAddV2+dense_714/kernel/Regularizer/Const:output:0$dense_714/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_714/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_714_1123065*
_output_shapes

:ii*
dtype0
#dense_714/kernel/Regularizer/SquareSquare:dense_714/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_714/kernel/Regularizer/Sum_1Sum'dense_714/kernel/Regularizer/Square:y:0-dense_714/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_714/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_714/kernel/Regularizer/mul_1Mul-dense_714/kernel/Regularizer/mul_1/x:output:0+dense_714/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_714/kernel/Regularizer/add_1AddV2$dense_714/kernel/Regularizer/add:z:0&dense_714/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_715/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_715/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_715_1123080*
_output_shapes

:ii*
dtype0
 dense_715/kernel/Regularizer/AbsAbs7dense_715/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_715/kernel/Regularizer/SumSum$dense_715/kernel/Regularizer/Abs:y:0-dense_715/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_715/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_715/kernel/Regularizer/mulMul+dense_715/kernel/Regularizer/mul/x:output:0)dense_715/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_715/kernel/Regularizer/addAddV2+dense_715/kernel/Regularizer/Const:output:0$dense_715/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_715/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_715_1123080*
_output_shapes

:ii*
dtype0
#dense_715/kernel/Regularizer/SquareSquare:dense_715/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_715/kernel/Regularizer/Sum_1Sum'dense_715/kernel/Regularizer/Square:y:0-dense_715/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_715/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_715/kernel/Regularizer/mul_1Mul-dense_715/kernel/Regularizer/mul_1/x:output:0+dense_715/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_715/kernel/Regularizer/add_1AddV2$dense_715/kernel/Regularizer/add:z:0&dense_715/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_716/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_716/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_716_1123095*
_output_shapes

:i=*
dtype0
 dense_716/kernel/Regularizer/AbsAbs7dense_716/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_716/kernel/Regularizer/SumSum$dense_716/kernel/Regularizer/Abs:y:0-dense_716/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_716/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_716/kernel/Regularizer/mulMul+dense_716/kernel/Regularizer/mul/x:output:0)dense_716/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_716/kernel/Regularizer/addAddV2+dense_716/kernel/Regularizer/Const:output:0$dense_716/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_716/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_716_1123095*
_output_shapes

:i=*
dtype0
#dense_716/kernel/Regularizer/SquareSquare:dense_716/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_716/kernel/Regularizer/Sum_1Sum'dense_716/kernel/Regularizer/Square:y:0-dense_716/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_716/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_716/kernel/Regularizer/mul_1Mul-dense_716/kernel/Regularizer/mul_1/x:output:0+dense_716/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_716/kernel/Regularizer/add_1AddV2$dense_716/kernel/Regularizer/add:z:0&dense_716/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_717/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_717/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_717_1123110*
_output_shapes

:==*
dtype0
 dense_717/kernel/Regularizer/AbsAbs7dense_717/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_717/kernel/Regularizer/SumSum$dense_717/kernel/Regularizer/Abs:y:0-dense_717/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_717/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_717/kernel/Regularizer/mulMul+dense_717/kernel/Regularizer/mul/x:output:0)dense_717/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_717/kernel/Regularizer/addAddV2+dense_717/kernel/Regularizer/Const:output:0$dense_717/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_717/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_717_1123110*
_output_shapes

:==*
dtype0
#dense_717/kernel/Regularizer/SquareSquare:dense_717/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_717/kernel/Regularizer/Sum_1Sum'dense_717/kernel/Regularizer/Square:y:0-dense_717/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_717/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_717/kernel/Regularizer/mul_1Mul-dense_717/kernel/Regularizer/mul_1/x:output:0+dense_717/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_717/kernel/Regularizer/add_1AddV2$dense_717/kernel/Regularizer/add:z:0&dense_717/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_718/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_718/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_718_1123125*
_output_shapes

:=7*
dtype0
 dense_718/kernel/Regularizer/AbsAbs7dense_718/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_718/kernel/Regularizer/SumSum$dense_718/kernel/Regularizer/Abs:y:0-dense_718/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_718/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_718/kernel/Regularizer/mulMul+dense_718/kernel/Regularizer/mul/x:output:0)dense_718/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_718/kernel/Regularizer/addAddV2+dense_718/kernel/Regularizer/Const:output:0$dense_718/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_718/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_718_1123125*
_output_shapes

:=7*
dtype0
#dense_718/kernel/Regularizer/SquareSquare:dense_718/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_718/kernel/Regularizer/Sum_1Sum'dense_718/kernel/Regularizer/Square:y:0-dense_718/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_718/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_718/kernel/Regularizer/mul_1Mul-dense_718/kernel/Regularizer/mul_1/x:output:0+dense_718/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_718/kernel/Regularizer/add_1AddV2$dense_718/kernel/Regularizer/add:z:0&dense_718/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_719/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_719/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_719_1123140*
_output_shapes

:77*
dtype0
 dense_719/kernel/Regularizer/AbsAbs7dense_719/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_719/kernel/Regularizer/SumSum$dense_719/kernel/Regularizer/Abs:y:0-dense_719/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_719/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_719/kernel/Regularizer/mulMul+dense_719/kernel/Regularizer/mul/x:output:0)dense_719/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_719/kernel/Regularizer/addAddV2+dense_719/kernel/Regularizer/Const:output:0$dense_719/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_719/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_719_1123140*
_output_shapes

:77*
dtype0
#dense_719/kernel/Regularizer/SquareSquare:dense_719/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_719/kernel/Regularizer/Sum_1Sum'dense_719/kernel/Regularizer/Square:y:0-dense_719/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_719/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_719/kernel/Regularizer/mul_1Mul-dense_719/kernel/Regularizer/mul_1/x:output:0+dense_719/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_719/kernel/Regularizer/add_1AddV2$dense_719/kernel/Regularizer/add:z:0&dense_719/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_720/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_640/StatefulPartitionedCall0^batch_normalization_641/StatefulPartitionedCall0^batch_normalization_642/StatefulPartitionedCall0^batch_normalization_643/StatefulPartitionedCall0^batch_normalization_644/StatefulPartitionedCall0^batch_normalization_645/StatefulPartitionedCall0^batch_normalization_646/StatefulPartitionedCall0^batch_normalization_647/StatefulPartitionedCall0^batch_normalization_648/StatefulPartitionedCall"^dense_711/StatefulPartitionedCall0^dense_711/kernel/Regularizer/Abs/ReadVariableOp3^dense_711/kernel/Regularizer/Square/ReadVariableOp"^dense_712/StatefulPartitionedCall0^dense_712/kernel/Regularizer/Abs/ReadVariableOp3^dense_712/kernel/Regularizer/Square/ReadVariableOp"^dense_713/StatefulPartitionedCall0^dense_713/kernel/Regularizer/Abs/ReadVariableOp3^dense_713/kernel/Regularizer/Square/ReadVariableOp"^dense_714/StatefulPartitionedCall0^dense_714/kernel/Regularizer/Abs/ReadVariableOp3^dense_714/kernel/Regularizer/Square/ReadVariableOp"^dense_715/StatefulPartitionedCall0^dense_715/kernel/Regularizer/Abs/ReadVariableOp3^dense_715/kernel/Regularizer/Square/ReadVariableOp"^dense_716/StatefulPartitionedCall0^dense_716/kernel/Regularizer/Abs/ReadVariableOp3^dense_716/kernel/Regularizer/Square/ReadVariableOp"^dense_717/StatefulPartitionedCall0^dense_717/kernel/Regularizer/Abs/ReadVariableOp3^dense_717/kernel/Regularizer/Square/ReadVariableOp"^dense_718/StatefulPartitionedCall0^dense_718/kernel/Regularizer/Abs/ReadVariableOp3^dense_718/kernel/Regularizer/Square/ReadVariableOp"^dense_719/StatefulPartitionedCall0^dense_719/kernel/Regularizer/Abs/ReadVariableOp3^dense_719/kernel/Regularizer/Square/ReadVariableOp"^dense_720/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_640/StatefulPartitionedCall/batch_normalization_640/StatefulPartitionedCall2b
/batch_normalization_641/StatefulPartitionedCall/batch_normalization_641/StatefulPartitionedCall2b
/batch_normalization_642/StatefulPartitionedCall/batch_normalization_642/StatefulPartitionedCall2b
/batch_normalization_643/StatefulPartitionedCall/batch_normalization_643/StatefulPartitionedCall2b
/batch_normalization_644/StatefulPartitionedCall/batch_normalization_644/StatefulPartitionedCall2b
/batch_normalization_645/StatefulPartitionedCall/batch_normalization_645/StatefulPartitionedCall2b
/batch_normalization_646/StatefulPartitionedCall/batch_normalization_646/StatefulPartitionedCall2b
/batch_normalization_647/StatefulPartitionedCall/batch_normalization_647/StatefulPartitionedCall2b
/batch_normalization_648/StatefulPartitionedCall/batch_normalization_648/StatefulPartitionedCall2F
!dense_711/StatefulPartitionedCall!dense_711/StatefulPartitionedCall2b
/dense_711/kernel/Regularizer/Abs/ReadVariableOp/dense_711/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_711/kernel/Regularizer/Square/ReadVariableOp2dense_711/kernel/Regularizer/Square/ReadVariableOp2F
!dense_712/StatefulPartitionedCall!dense_712/StatefulPartitionedCall2b
/dense_712/kernel/Regularizer/Abs/ReadVariableOp/dense_712/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_712/kernel/Regularizer/Square/ReadVariableOp2dense_712/kernel/Regularizer/Square/ReadVariableOp2F
!dense_713/StatefulPartitionedCall!dense_713/StatefulPartitionedCall2b
/dense_713/kernel/Regularizer/Abs/ReadVariableOp/dense_713/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_713/kernel/Regularizer/Square/ReadVariableOp2dense_713/kernel/Regularizer/Square/ReadVariableOp2F
!dense_714/StatefulPartitionedCall!dense_714/StatefulPartitionedCall2b
/dense_714/kernel/Regularizer/Abs/ReadVariableOp/dense_714/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_714/kernel/Regularizer/Square/ReadVariableOp2dense_714/kernel/Regularizer/Square/ReadVariableOp2F
!dense_715/StatefulPartitionedCall!dense_715/StatefulPartitionedCall2b
/dense_715/kernel/Regularizer/Abs/ReadVariableOp/dense_715/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_715/kernel/Regularizer/Square/ReadVariableOp2dense_715/kernel/Regularizer/Square/ReadVariableOp2F
!dense_716/StatefulPartitionedCall!dense_716/StatefulPartitionedCall2b
/dense_716/kernel/Regularizer/Abs/ReadVariableOp/dense_716/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_716/kernel/Regularizer/Square/ReadVariableOp2dense_716/kernel/Regularizer/Square/ReadVariableOp2F
!dense_717/StatefulPartitionedCall!dense_717/StatefulPartitionedCall2b
/dense_717/kernel/Regularizer/Abs/ReadVariableOp/dense_717/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_717/kernel/Regularizer/Square/ReadVariableOp2dense_717/kernel/Regularizer/Square/ReadVariableOp2F
!dense_718/StatefulPartitionedCall!dense_718/StatefulPartitionedCall2b
/dense_718/kernel/Regularizer/Abs/ReadVariableOp/dense_718/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_718/kernel/Regularizer/Square/ReadVariableOp2dense_718/kernel/Regularizer/Square/ReadVariableOp2F
!dense_719/StatefulPartitionedCall!dense_719/StatefulPartitionedCall2b
/dense_719/kernel/Regularizer/Abs/ReadVariableOp/dense_719/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_719/kernel/Regularizer/Square/ReadVariableOp2dense_719/kernel/Regularizer/Square/ReadVariableOp2F
!dense_720/StatefulPartitionedCall!dense_720/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_71_input:$ 

_output_shapes

::$ 

_output_shapes

:
­
M
1__inference_leaky_re_lu_644_layer_call_fn_1125667

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
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_644_layer_call_and_return_conditional_losses_1121746`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿi:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs

ã
__inference_loss_fn_6_1126387J
8dense_717_kernel_regularizer_abs_readvariableop_resource:==
identity¢/dense_717/kernel/Regularizer/Abs/ReadVariableOp¢2dense_717/kernel/Regularizer/Square/ReadVariableOpg
"dense_717/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
/dense_717/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8dense_717_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:==*
dtype0
 dense_717/kernel/Regularizer/AbsAbs7dense_717/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_717/kernel/Regularizer/SumSum$dense_717/kernel/Regularizer/Abs:y:0-dense_717/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_717/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_717/kernel/Regularizer/mulMul+dense_717/kernel/Regularizer/mul/x:output:0)dense_717/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_717/kernel/Regularizer/addAddV2+dense_717/kernel/Regularizer/Const:output:0$dense_717/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: «
2dense_717/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense_717_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:==*
dtype0
#dense_717/kernel/Regularizer/SquareSquare:dense_717/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_717/kernel/Regularizer/Sum_1Sum'dense_717/kernel/Regularizer/Square:y:0-dense_717/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_717/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_717/kernel/Regularizer/mul_1Mul-dense_717/kernel/Regularizer/mul_1/x:output:0+dense_717/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_717/kernel/Regularizer/add_1AddV2$dense_717/kernel/Regularizer/add:z:0&dense_717/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
IdentityIdentity&dense_717/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ­
NoOpNoOp0^dense_717/kernel/Regularizer/Abs/ReadVariableOp3^dense_717/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_717/kernel/Regularizer/Abs/ReadVariableOp/dense_717/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_717/kernel/Regularizer/Square/ReadVariableOp2dense_717/kernel/Regularizer/Square/ReadVariableOp
Ç
Ó!
J__inference_sequential_71_layer_call_and_return_conditional_losses_1123582
normalization_71_input
normalization_71_sub_y
normalization_71_sqrt_x#
dense_711_1123306:i
dense_711_1123308:i-
batch_normalization_640_1123311:i-
batch_normalization_640_1123313:i-
batch_normalization_640_1123315:i-
batch_normalization_640_1123317:i#
dense_712_1123321:ii
dense_712_1123323:i-
batch_normalization_641_1123326:i-
batch_normalization_641_1123328:i-
batch_normalization_641_1123330:i-
batch_normalization_641_1123332:i#
dense_713_1123336:ii
dense_713_1123338:i-
batch_normalization_642_1123341:i-
batch_normalization_642_1123343:i-
batch_normalization_642_1123345:i-
batch_normalization_642_1123347:i#
dense_714_1123351:ii
dense_714_1123353:i-
batch_normalization_643_1123356:i-
batch_normalization_643_1123358:i-
batch_normalization_643_1123360:i-
batch_normalization_643_1123362:i#
dense_715_1123366:ii
dense_715_1123368:i-
batch_normalization_644_1123371:i-
batch_normalization_644_1123373:i-
batch_normalization_644_1123375:i-
batch_normalization_644_1123377:i#
dense_716_1123381:i=
dense_716_1123383:=-
batch_normalization_645_1123386:=-
batch_normalization_645_1123388:=-
batch_normalization_645_1123390:=-
batch_normalization_645_1123392:=#
dense_717_1123396:==
dense_717_1123398:=-
batch_normalization_646_1123401:=-
batch_normalization_646_1123403:=-
batch_normalization_646_1123405:=-
batch_normalization_646_1123407:=#
dense_718_1123411:=7
dense_718_1123413:7-
batch_normalization_647_1123416:7-
batch_normalization_647_1123418:7-
batch_normalization_647_1123420:7-
batch_normalization_647_1123422:7#
dense_719_1123426:77
dense_719_1123428:7-
batch_normalization_648_1123431:7-
batch_normalization_648_1123433:7-
batch_normalization_648_1123435:7-
batch_normalization_648_1123437:7#
dense_720_1123441:7
dense_720_1123443:
identity¢/batch_normalization_640/StatefulPartitionedCall¢/batch_normalization_641/StatefulPartitionedCall¢/batch_normalization_642/StatefulPartitionedCall¢/batch_normalization_643/StatefulPartitionedCall¢/batch_normalization_644/StatefulPartitionedCall¢/batch_normalization_645/StatefulPartitionedCall¢/batch_normalization_646/StatefulPartitionedCall¢/batch_normalization_647/StatefulPartitionedCall¢/batch_normalization_648/StatefulPartitionedCall¢!dense_711/StatefulPartitionedCall¢/dense_711/kernel/Regularizer/Abs/ReadVariableOp¢2dense_711/kernel/Regularizer/Square/ReadVariableOp¢!dense_712/StatefulPartitionedCall¢/dense_712/kernel/Regularizer/Abs/ReadVariableOp¢2dense_712/kernel/Regularizer/Square/ReadVariableOp¢!dense_713/StatefulPartitionedCall¢/dense_713/kernel/Regularizer/Abs/ReadVariableOp¢2dense_713/kernel/Regularizer/Square/ReadVariableOp¢!dense_714/StatefulPartitionedCall¢/dense_714/kernel/Regularizer/Abs/ReadVariableOp¢2dense_714/kernel/Regularizer/Square/ReadVariableOp¢!dense_715/StatefulPartitionedCall¢/dense_715/kernel/Regularizer/Abs/ReadVariableOp¢2dense_715/kernel/Regularizer/Square/ReadVariableOp¢!dense_716/StatefulPartitionedCall¢/dense_716/kernel/Regularizer/Abs/ReadVariableOp¢2dense_716/kernel/Regularizer/Square/ReadVariableOp¢!dense_717/StatefulPartitionedCall¢/dense_717/kernel/Regularizer/Abs/ReadVariableOp¢2dense_717/kernel/Regularizer/Square/ReadVariableOp¢!dense_718/StatefulPartitionedCall¢/dense_718/kernel/Regularizer/Abs/ReadVariableOp¢2dense_718/kernel/Regularizer/Square/ReadVariableOp¢!dense_719/StatefulPartitionedCall¢/dense_719/kernel/Regularizer/Abs/ReadVariableOp¢2dense_719/kernel/Regularizer/Square/ReadVariableOp¢!dense_720/StatefulPartitionedCall}
normalization_71/subSubnormalization_71_inputnormalization_71_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_71/SqrtSqrtnormalization_71_sqrt_x*
T0*
_output_shapes

:_
normalization_71/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_71/MaximumMaximumnormalization_71/Sqrt:y:0#normalization_71/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_71/truedivRealDivnormalization_71/sub:z:0normalization_71/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_711/StatefulPartitionedCallStatefulPartitionedCallnormalization_71/truediv:z:0dense_711_1123306dense_711_1123308*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_711_layer_call_and_return_conditional_losses_1121538
/batch_normalization_640/StatefulPartitionedCallStatefulPartitionedCall*dense_711/StatefulPartitionedCall:output:0batch_normalization_640_1123311batch_normalization_640_1123313batch_normalization_640_1123315batch_normalization_640_1123317*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_640_layer_call_and_return_conditional_losses_1120832ù
leaky_re_lu_640/PartitionedCallPartitionedCall8batch_normalization_640/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_640_layer_call_and_return_conditional_losses_1121558
!dense_712/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_640/PartitionedCall:output:0dense_712_1123321dense_712_1123323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_712_layer_call_and_return_conditional_losses_1121585
/batch_normalization_641/StatefulPartitionedCallStatefulPartitionedCall*dense_712/StatefulPartitionedCall:output:0batch_normalization_641_1123326batch_normalization_641_1123328batch_normalization_641_1123330batch_normalization_641_1123332*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_641_layer_call_and_return_conditional_losses_1120914ù
leaky_re_lu_641/PartitionedCallPartitionedCall8batch_normalization_641/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_641_layer_call_and_return_conditional_losses_1121605
!dense_713/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_641/PartitionedCall:output:0dense_713_1123336dense_713_1123338*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_713_layer_call_and_return_conditional_losses_1121632
/batch_normalization_642/StatefulPartitionedCallStatefulPartitionedCall*dense_713/StatefulPartitionedCall:output:0batch_normalization_642_1123341batch_normalization_642_1123343batch_normalization_642_1123345batch_normalization_642_1123347*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_642_layer_call_and_return_conditional_losses_1120996ù
leaky_re_lu_642/PartitionedCallPartitionedCall8batch_normalization_642/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_642_layer_call_and_return_conditional_losses_1121652
!dense_714/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_642/PartitionedCall:output:0dense_714_1123351dense_714_1123353*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_714_layer_call_and_return_conditional_losses_1121679
/batch_normalization_643/StatefulPartitionedCallStatefulPartitionedCall*dense_714/StatefulPartitionedCall:output:0batch_normalization_643_1123356batch_normalization_643_1123358batch_normalization_643_1123360batch_normalization_643_1123362*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_643_layer_call_and_return_conditional_losses_1121078ù
leaky_re_lu_643/PartitionedCallPartitionedCall8batch_normalization_643/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_643_layer_call_and_return_conditional_losses_1121699
!dense_715/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_643/PartitionedCall:output:0dense_715_1123366dense_715_1123368*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_715_layer_call_and_return_conditional_losses_1121726
/batch_normalization_644/StatefulPartitionedCallStatefulPartitionedCall*dense_715/StatefulPartitionedCall:output:0batch_normalization_644_1123371batch_normalization_644_1123373batch_normalization_644_1123375batch_normalization_644_1123377*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_644_layer_call_and_return_conditional_losses_1121160ù
leaky_re_lu_644/PartitionedCallPartitionedCall8batch_normalization_644/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_644_layer_call_and_return_conditional_losses_1121746
!dense_716/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_644/PartitionedCall:output:0dense_716_1123381dense_716_1123383*
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
GPU 2J 8 *O
fJRH
F__inference_dense_716_layer_call_and_return_conditional_losses_1121773
/batch_normalization_645/StatefulPartitionedCallStatefulPartitionedCall*dense_716/StatefulPartitionedCall:output:0batch_normalization_645_1123386batch_normalization_645_1123388batch_normalization_645_1123390batch_normalization_645_1123392*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_645_layer_call_and_return_conditional_losses_1121242ù
leaky_re_lu_645/PartitionedCallPartitionedCall8batch_normalization_645/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_645_layer_call_and_return_conditional_losses_1121793
!dense_717/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_645/PartitionedCall:output:0dense_717_1123396dense_717_1123398*
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
GPU 2J 8 *O
fJRH
F__inference_dense_717_layer_call_and_return_conditional_losses_1121820
/batch_normalization_646/StatefulPartitionedCallStatefulPartitionedCall*dense_717/StatefulPartitionedCall:output:0batch_normalization_646_1123401batch_normalization_646_1123403batch_normalization_646_1123405batch_normalization_646_1123407*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_646_layer_call_and_return_conditional_losses_1121324ù
leaky_re_lu_646/PartitionedCallPartitionedCall8batch_normalization_646/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_646_layer_call_and_return_conditional_losses_1121840
!dense_718/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_646/PartitionedCall:output:0dense_718_1123411dense_718_1123413*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_718_layer_call_and_return_conditional_losses_1121867
/batch_normalization_647/StatefulPartitionedCallStatefulPartitionedCall*dense_718/StatefulPartitionedCall:output:0batch_normalization_647_1123416batch_normalization_647_1123418batch_normalization_647_1123420batch_normalization_647_1123422*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_647_layer_call_and_return_conditional_losses_1121406ù
leaky_re_lu_647/PartitionedCallPartitionedCall8batch_normalization_647/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_647_layer_call_and_return_conditional_losses_1121887
!dense_719/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_647/PartitionedCall:output:0dense_719_1123426dense_719_1123428*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_719_layer_call_and_return_conditional_losses_1121914
/batch_normalization_648/StatefulPartitionedCallStatefulPartitionedCall*dense_719/StatefulPartitionedCall:output:0batch_normalization_648_1123431batch_normalization_648_1123433batch_normalization_648_1123435batch_normalization_648_1123437*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_648_layer_call_and_return_conditional_losses_1121488ù
leaky_re_lu_648/PartitionedCallPartitionedCall8batch_normalization_648/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_648_layer_call_and_return_conditional_losses_1121934
!dense_720/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_648/PartitionedCall:output:0dense_720_1123441dense_720_1123443*
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
F__inference_dense_720_layer_call_and_return_conditional_losses_1121946g
"dense_711/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_711/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_711_1123306*
_output_shapes

:i*
dtype0
 dense_711/kernel/Regularizer/AbsAbs7dense_711/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_711/kernel/Regularizer/SumSum$dense_711/kernel/Regularizer/Abs:y:0-dense_711/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_711/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_711/kernel/Regularizer/mulMul+dense_711/kernel/Regularizer/mul/x:output:0)dense_711/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_711/kernel/Regularizer/addAddV2+dense_711/kernel/Regularizer/Const:output:0$dense_711/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_711/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_711_1123306*
_output_shapes

:i*
dtype0
#dense_711/kernel/Regularizer/SquareSquare:dense_711/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_711/kernel/Regularizer/Sum_1Sum'dense_711/kernel/Regularizer/Square:y:0-dense_711/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_711/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_711/kernel/Regularizer/mul_1Mul-dense_711/kernel/Regularizer/mul_1/x:output:0+dense_711/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_711/kernel/Regularizer/add_1AddV2$dense_711/kernel/Regularizer/add:z:0&dense_711/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_712/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_712/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_712_1123321*
_output_shapes

:ii*
dtype0
 dense_712/kernel/Regularizer/AbsAbs7dense_712/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_712/kernel/Regularizer/SumSum$dense_712/kernel/Regularizer/Abs:y:0-dense_712/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_712/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_712/kernel/Regularizer/mulMul+dense_712/kernel/Regularizer/mul/x:output:0)dense_712/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_712/kernel/Regularizer/addAddV2+dense_712/kernel/Regularizer/Const:output:0$dense_712/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_712/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_712_1123321*
_output_shapes

:ii*
dtype0
#dense_712/kernel/Regularizer/SquareSquare:dense_712/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_712/kernel/Regularizer/Sum_1Sum'dense_712/kernel/Regularizer/Square:y:0-dense_712/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_712/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_712/kernel/Regularizer/mul_1Mul-dense_712/kernel/Regularizer/mul_1/x:output:0+dense_712/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_712/kernel/Regularizer/add_1AddV2$dense_712/kernel/Regularizer/add:z:0&dense_712/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_713/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_713/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_713_1123336*
_output_shapes

:ii*
dtype0
 dense_713/kernel/Regularizer/AbsAbs7dense_713/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_713/kernel/Regularizer/SumSum$dense_713/kernel/Regularizer/Abs:y:0-dense_713/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_713/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_713/kernel/Regularizer/mulMul+dense_713/kernel/Regularizer/mul/x:output:0)dense_713/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_713/kernel/Regularizer/addAddV2+dense_713/kernel/Regularizer/Const:output:0$dense_713/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_713/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_713_1123336*
_output_shapes

:ii*
dtype0
#dense_713/kernel/Regularizer/SquareSquare:dense_713/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_713/kernel/Regularizer/Sum_1Sum'dense_713/kernel/Regularizer/Square:y:0-dense_713/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_713/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_713/kernel/Regularizer/mul_1Mul-dense_713/kernel/Regularizer/mul_1/x:output:0+dense_713/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_713/kernel/Regularizer/add_1AddV2$dense_713/kernel/Regularizer/add:z:0&dense_713/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_714/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_714/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_714_1123351*
_output_shapes

:ii*
dtype0
 dense_714/kernel/Regularizer/AbsAbs7dense_714/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_714/kernel/Regularizer/SumSum$dense_714/kernel/Regularizer/Abs:y:0-dense_714/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_714/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_714/kernel/Regularizer/mulMul+dense_714/kernel/Regularizer/mul/x:output:0)dense_714/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_714/kernel/Regularizer/addAddV2+dense_714/kernel/Regularizer/Const:output:0$dense_714/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_714/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_714_1123351*
_output_shapes

:ii*
dtype0
#dense_714/kernel/Regularizer/SquareSquare:dense_714/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_714/kernel/Regularizer/Sum_1Sum'dense_714/kernel/Regularizer/Square:y:0-dense_714/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_714/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_714/kernel/Regularizer/mul_1Mul-dense_714/kernel/Regularizer/mul_1/x:output:0+dense_714/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_714/kernel/Regularizer/add_1AddV2$dense_714/kernel/Regularizer/add:z:0&dense_714/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_715/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_715/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_715_1123366*
_output_shapes

:ii*
dtype0
 dense_715/kernel/Regularizer/AbsAbs7dense_715/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_715/kernel/Regularizer/SumSum$dense_715/kernel/Regularizer/Abs:y:0-dense_715/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_715/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_715/kernel/Regularizer/mulMul+dense_715/kernel/Regularizer/mul/x:output:0)dense_715/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_715/kernel/Regularizer/addAddV2+dense_715/kernel/Regularizer/Const:output:0$dense_715/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_715/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_715_1123366*
_output_shapes

:ii*
dtype0
#dense_715/kernel/Regularizer/SquareSquare:dense_715/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_715/kernel/Regularizer/Sum_1Sum'dense_715/kernel/Regularizer/Square:y:0-dense_715/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_715/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_715/kernel/Regularizer/mul_1Mul-dense_715/kernel/Regularizer/mul_1/x:output:0+dense_715/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_715/kernel/Regularizer/add_1AddV2$dense_715/kernel/Regularizer/add:z:0&dense_715/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_716/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_716/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_716_1123381*
_output_shapes

:i=*
dtype0
 dense_716/kernel/Regularizer/AbsAbs7dense_716/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_716/kernel/Regularizer/SumSum$dense_716/kernel/Regularizer/Abs:y:0-dense_716/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_716/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_716/kernel/Regularizer/mulMul+dense_716/kernel/Regularizer/mul/x:output:0)dense_716/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_716/kernel/Regularizer/addAddV2+dense_716/kernel/Regularizer/Const:output:0$dense_716/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_716/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_716_1123381*
_output_shapes

:i=*
dtype0
#dense_716/kernel/Regularizer/SquareSquare:dense_716/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_716/kernel/Regularizer/Sum_1Sum'dense_716/kernel/Regularizer/Square:y:0-dense_716/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_716/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_716/kernel/Regularizer/mul_1Mul-dense_716/kernel/Regularizer/mul_1/x:output:0+dense_716/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_716/kernel/Regularizer/add_1AddV2$dense_716/kernel/Regularizer/add:z:0&dense_716/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_717/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_717/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_717_1123396*
_output_shapes

:==*
dtype0
 dense_717/kernel/Regularizer/AbsAbs7dense_717/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_717/kernel/Regularizer/SumSum$dense_717/kernel/Regularizer/Abs:y:0-dense_717/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_717/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_717/kernel/Regularizer/mulMul+dense_717/kernel/Regularizer/mul/x:output:0)dense_717/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_717/kernel/Regularizer/addAddV2+dense_717/kernel/Regularizer/Const:output:0$dense_717/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_717/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_717_1123396*
_output_shapes

:==*
dtype0
#dense_717/kernel/Regularizer/SquareSquare:dense_717/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_717/kernel/Regularizer/Sum_1Sum'dense_717/kernel/Regularizer/Square:y:0-dense_717/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_717/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_717/kernel/Regularizer/mul_1Mul-dense_717/kernel/Regularizer/mul_1/x:output:0+dense_717/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_717/kernel/Regularizer/add_1AddV2$dense_717/kernel/Regularizer/add:z:0&dense_717/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_718/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_718/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_718_1123411*
_output_shapes

:=7*
dtype0
 dense_718/kernel/Regularizer/AbsAbs7dense_718/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_718/kernel/Regularizer/SumSum$dense_718/kernel/Regularizer/Abs:y:0-dense_718/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_718/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_718/kernel/Regularizer/mulMul+dense_718/kernel/Regularizer/mul/x:output:0)dense_718/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_718/kernel/Regularizer/addAddV2+dense_718/kernel/Regularizer/Const:output:0$dense_718/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_718/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_718_1123411*
_output_shapes

:=7*
dtype0
#dense_718/kernel/Regularizer/SquareSquare:dense_718/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_718/kernel/Regularizer/Sum_1Sum'dense_718/kernel/Regularizer/Square:y:0-dense_718/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_718/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_718/kernel/Regularizer/mul_1Mul-dense_718/kernel/Regularizer/mul_1/x:output:0+dense_718/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_718/kernel/Regularizer/add_1AddV2$dense_718/kernel/Regularizer/add:z:0&dense_718/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_719/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_719/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_719_1123426*
_output_shapes

:77*
dtype0
 dense_719/kernel/Regularizer/AbsAbs7dense_719/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_719/kernel/Regularizer/SumSum$dense_719/kernel/Regularizer/Abs:y:0-dense_719/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_719/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_719/kernel/Regularizer/mulMul+dense_719/kernel/Regularizer/mul/x:output:0)dense_719/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_719/kernel/Regularizer/addAddV2+dense_719/kernel/Regularizer/Const:output:0$dense_719/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_719/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_719_1123426*
_output_shapes

:77*
dtype0
#dense_719/kernel/Regularizer/SquareSquare:dense_719/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_719/kernel/Regularizer/Sum_1Sum'dense_719/kernel/Regularizer/Square:y:0-dense_719/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_719/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_719/kernel/Regularizer/mul_1Mul-dense_719/kernel/Regularizer/mul_1/x:output:0+dense_719/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_719/kernel/Regularizer/add_1AddV2$dense_719/kernel/Regularizer/add:z:0&dense_719/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_720/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_640/StatefulPartitionedCall0^batch_normalization_641/StatefulPartitionedCall0^batch_normalization_642/StatefulPartitionedCall0^batch_normalization_643/StatefulPartitionedCall0^batch_normalization_644/StatefulPartitionedCall0^batch_normalization_645/StatefulPartitionedCall0^batch_normalization_646/StatefulPartitionedCall0^batch_normalization_647/StatefulPartitionedCall0^batch_normalization_648/StatefulPartitionedCall"^dense_711/StatefulPartitionedCall0^dense_711/kernel/Regularizer/Abs/ReadVariableOp3^dense_711/kernel/Regularizer/Square/ReadVariableOp"^dense_712/StatefulPartitionedCall0^dense_712/kernel/Regularizer/Abs/ReadVariableOp3^dense_712/kernel/Regularizer/Square/ReadVariableOp"^dense_713/StatefulPartitionedCall0^dense_713/kernel/Regularizer/Abs/ReadVariableOp3^dense_713/kernel/Regularizer/Square/ReadVariableOp"^dense_714/StatefulPartitionedCall0^dense_714/kernel/Regularizer/Abs/ReadVariableOp3^dense_714/kernel/Regularizer/Square/ReadVariableOp"^dense_715/StatefulPartitionedCall0^dense_715/kernel/Regularizer/Abs/ReadVariableOp3^dense_715/kernel/Regularizer/Square/ReadVariableOp"^dense_716/StatefulPartitionedCall0^dense_716/kernel/Regularizer/Abs/ReadVariableOp3^dense_716/kernel/Regularizer/Square/ReadVariableOp"^dense_717/StatefulPartitionedCall0^dense_717/kernel/Regularizer/Abs/ReadVariableOp3^dense_717/kernel/Regularizer/Square/ReadVariableOp"^dense_718/StatefulPartitionedCall0^dense_718/kernel/Regularizer/Abs/ReadVariableOp3^dense_718/kernel/Regularizer/Square/ReadVariableOp"^dense_719/StatefulPartitionedCall0^dense_719/kernel/Regularizer/Abs/ReadVariableOp3^dense_719/kernel/Regularizer/Square/ReadVariableOp"^dense_720/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_640/StatefulPartitionedCall/batch_normalization_640/StatefulPartitionedCall2b
/batch_normalization_641/StatefulPartitionedCall/batch_normalization_641/StatefulPartitionedCall2b
/batch_normalization_642/StatefulPartitionedCall/batch_normalization_642/StatefulPartitionedCall2b
/batch_normalization_643/StatefulPartitionedCall/batch_normalization_643/StatefulPartitionedCall2b
/batch_normalization_644/StatefulPartitionedCall/batch_normalization_644/StatefulPartitionedCall2b
/batch_normalization_645/StatefulPartitionedCall/batch_normalization_645/StatefulPartitionedCall2b
/batch_normalization_646/StatefulPartitionedCall/batch_normalization_646/StatefulPartitionedCall2b
/batch_normalization_647/StatefulPartitionedCall/batch_normalization_647/StatefulPartitionedCall2b
/batch_normalization_648/StatefulPartitionedCall/batch_normalization_648/StatefulPartitionedCall2F
!dense_711/StatefulPartitionedCall!dense_711/StatefulPartitionedCall2b
/dense_711/kernel/Regularizer/Abs/ReadVariableOp/dense_711/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_711/kernel/Regularizer/Square/ReadVariableOp2dense_711/kernel/Regularizer/Square/ReadVariableOp2F
!dense_712/StatefulPartitionedCall!dense_712/StatefulPartitionedCall2b
/dense_712/kernel/Regularizer/Abs/ReadVariableOp/dense_712/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_712/kernel/Regularizer/Square/ReadVariableOp2dense_712/kernel/Regularizer/Square/ReadVariableOp2F
!dense_713/StatefulPartitionedCall!dense_713/StatefulPartitionedCall2b
/dense_713/kernel/Regularizer/Abs/ReadVariableOp/dense_713/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_713/kernel/Regularizer/Square/ReadVariableOp2dense_713/kernel/Regularizer/Square/ReadVariableOp2F
!dense_714/StatefulPartitionedCall!dense_714/StatefulPartitionedCall2b
/dense_714/kernel/Regularizer/Abs/ReadVariableOp/dense_714/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_714/kernel/Regularizer/Square/ReadVariableOp2dense_714/kernel/Regularizer/Square/ReadVariableOp2F
!dense_715/StatefulPartitionedCall!dense_715/StatefulPartitionedCall2b
/dense_715/kernel/Regularizer/Abs/ReadVariableOp/dense_715/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_715/kernel/Regularizer/Square/ReadVariableOp2dense_715/kernel/Regularizer/Square/ReadVariableOp2F
!dense_716/StatefulPartitionedCall!dense_716/StatefulPartitionedCall2b
/dense_716/kernel/Regularizer/Abs/ReadVariableOp/dense_716/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_716/kernel/Regularizer/Square/ReadVariableOp2dense_716/kernel/Regularizer/Square/ReadVariableOp2F
!dense_717/StatefulPartitionedCall!dense_717/StatefulPartitionedCall2b
/dense_717/kernel/Regularizer/Abs/ReadVariableOp/dense_717/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_717/kernel/Regularizer/Square/ReadVariableOp2dense_717/kernel/Regularizer/Square/ReadVariableOp2F
!dense_718/StatefulPartitionedCall!dense_718/StatefulPartitionedCall2b
/dense_718/kernel/Regularizer/Abs/ReadVariableOp/dense_718/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_718/kernel/Regularizer/Square/ReadVariableOp2dense_718/kernel/Regularizer/Square/ReadVariableOp2F
!dense_719/StatefulPartitionedCall!dense_719/StatefulPartitionedCall2b
/dense_719/kernel/Regularizer/Abs/ReadVariableOp/dense_719/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_719/kernel/Regularizer/Square/ReadVariableOp2dense_719/kernel/Regularizer/Square/ReadVariableOp2F
!dense_720/StatefulPartitionedCall!dense_720/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_71_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ñ
³
T__inference_batch_normalization_643_layer_call_and_return_conditional_losses_1125489

inputs/
!batchnorm_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i1
#batchnorm_readvariableop_1_resource:i1
#batchnorm_readvariableop_2_resource:i
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:iz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_645_layer_call_fn_1125806

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
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_645_layer_call_and_return_conditional_losses_1121793`
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
¢
@
"__inference__wrapped_model_1120761
normalization_71_input(
$sequential_71_normalization_71_sub_y)
%sequential_71_normalization_71_sqrt_xH
6sequential_71_dense_711_matmul_readvariableop_resource:iE
7sequential_71_dense_711_biasadd_readvariableop_resource:iU
Gsequential_71_batch_normalization_640_batchnorm_readvariableop_resource:iY
Ksequential_71_batch_normalization_640_batchnorm_mul_readvariableop_resource:iW
Isequential_71_batch_normalization_640_batchnorm_readvariableop_1_resource:iW
Isequential_71_batch_normalization_640_batchnorm_readvariableop_2_resource:iH
6sequential_71_dense_712_matmul_readvariableop_resource:iiE
7sequential_71_dense_712_biasadd_readvariableop_resource:iU
Gsequential_71_batch_normalization_641_batchnorm_readvariableop_resource:iY
Ksequential_71_batch_normalization_641_batchnorm_mul_readvariableop_resource:iW
Isequential_71_batch_normalization_641_batchnorm_readvariableop_1_resource:iW
Isequential_71_batch_normalization_641_batchnorm_readvariableop_2_resource:iH
6sequential_71_dense_713_matmul_readvariableop_resource:iiE
7sequential_71_dense_713_biasadd_readvariableop_resource:iU
Gsequential_71_batch_normalization_642_batchnorm_readvariableop_resource:iY
Ksequential_71_batch_normalization_642_batchnorm_mul_readvariableop_resource:iW
Isequential_71_batch_normalization_642_batchnorm_readvariableop_1_resource:iW
Isequential_71_batch_normalization_642_batchnorm_readvariableop_2_resource:iH
6sequential_71_dense_714_matmul_readvariableop_resource:iiE
7sequential_71_dense_714_biasadd_readvariableop_resource:iU
Gsequential_71_batch_normalization_643_batchnorm_readvariableop_resource:iY
Ksequential_71_batch_normalization_643_batchnorm_mul_readvariableop_resource:iW
Isequential_71_batch_normalization_643_batchnorm_readvariableop_1_resource:iW
Isequential_71_batch_normalization_643_batchnorm_readvariableop_2_resource:iH
6sequential_71_dense_715_matmul_readvariableop_resource:iiE
7sequential_71_dense_715_biasadd_readvariableop_resource:iU
Gsequential_71_batch_normalization_644_batchnorm_readvariableop_resource:iY
Ksequential_71_batch_normalization_644_batchnorm_mul_readvariableop_resource:iW
Isequential_71_batch_normalization_644_batchnorm_readvariableop_1_resource:iW
Isequential_71_batch_normalization_644_batchnorm_readvariableop_2_resource:iH
6sequential_71_dense_716_matmul_readvariableop_resource:i=E
7sequential_71_dense_716_biasadd_readvariableop_resource:=U
Gsequential_71_batch_normalization_645_batchnorm_readvariableop_resource:=Y
Ksequential_71_batch_normalization_645_batchnorm_mul_readvariableop_resource:=W
Isequential_71_batch_normalization_645_batchnorm_readvariableop_1_resource:=W
Isequential_71_batch_normalization_645_batchnorm_readvariableop_2_resource:=H
6sequential_71_dense_717_matmul_readvariableop_resource:==E
7sequential_71_dense_717_biasadd_readvariableop_resource:=U
Gsequential_71_batch_normalization_646_batchnorm_readvariableop_resource:=Y
Ksequential_71_batch_normalization_646_batchnorm_mul_readvariableop_resource:=W
Isequential_71_batch_normalization_646_batchnorm_readvariableop_1_resource:=W
Isequential_71_batch_normalization_646_batchnorm_readvariableop_2_resource:=H
6sequential_71_dense_718_matmul_readvariableop_resource:=7E
7sequential_71_dense_718_biasadd_readvariableop_resource:7U
Gsequential_71_batch_normalization_647_batchnorm_readvariableop_resource:7Y
Ksequential_71_batch_normalization_647_batchnorm_mul_readvariableop_resource:7W
Isequential_71_batch_normalization_647_batchnorm_readvariableop_1_resource:7W
Isequential_71_batch_normalization_647_batchnorm_readvariableop_2_resource:7H
6sequential_71_dense_719_matmul_readvariableop_resource:77E
7sequential_71_dense_719_biasadd_readvariableop_resource:7U
Gsequential_71_batch_normalization_648_batchnorm_readvariableop_resource:7Y
Ksequential_71_batch_normalization_648_batchnorm_mul_readvariableop_resource:7W
Isequential_71_batch_normalization_648_batchnorm_readvariableop_1_resource:7W
Isequential_71_batch_normalization_648_batchnorm_readvariableop_2_resource:7H
6sequential_71_dense_720_matmul_readvariableop_resource:7E
7sequential_71_dense_720_biasadd_readvariableop_resource:
identity¢>sequential_71/batch_normalization_640/batchnorm/ReadVariableOp¢@sequential_71/batch_normalization_640/batchnorm/ReadVariableOp_1¢@sequential_71/batch_normalization_640/batchnorm/ReadVariableOp_2¢Bsequential_71/batch_normalization_640/batchnorm/mul/ReadVariableOp¢>sequential_71/batch_normalization_641/batchnorm/ReadVariableOp¢@sequential_71/batch_normalization_641/batchnorm/ReadVariableOp_1¢@sequential_71/batch_normalization_641/batchnorm/ReadVariableOp_2¢Bsequential_71/batch_normalization_641/batchnorm/mul/ReadVariableOp¢>sequential_71/batch_normalization_642/batchnorm/ReadVariableOp¢@sequential_71/batch_normalization_642/batchnorm/ReadVariableOp_1¢@sequential_71/batch_normalization_642/batchnorm/ReadVariableOp_2¢Bsequential_71/batch_normalization_642/batchnorm/mul/ReadVariableOp¢>sequential_71/batch_normalization_643/batchnorm/ReadVariableOp¢@sequential_71/batch_normalization_643/batchnorm/ReadVariableOp_1¢@sequential_71/batch_normalization_643/batchnorm/ReadVariableOp_2¢Bsequential_71/batch_normalization_643/batchnorm/mul/ReadVariableOp¢>sequential_71/batch_normalization_644/batchnorm/ReadVariableOp¢@sequential_71/batch_normalization_644/batchnorm/ReadVariableOp_1¢@sequential_71/batch_normalization_644/batchnorm/ReadVariableOp_2¢Bsequential_71/batch_normalization_644/batchnorm/mul/ReadVariableOp¢>sequential_71/batch_normalization_645/batchnorm/ReadVariableOp¢@sequential_71/batch_normalization_645/batchnorm/ReadVariableOp_1¢@sequential_71/batch_normalization_645/batchnorm/ReadVariableOp_2¢Bsequential_71/batch_normalization_645/batchnorm/mul/ReadVariableOp¢>sequential_71/batch_normalization_646/batchnorm/ReadVariableOp¢@sequential_71/batch_normalization_646/batchnorm/ReadVariableOp_1¢@sequential_71/batch_normalization_646/batchnorm/ReadVariableOp_2¢Bsequential_71/batch_normalization_646/batchnorm/mul/ReadVariableOp¢>sequential_71/batch_normalization_647/batchnorm/ReadVariableOp¢@sequential_71/batch_normalization_647/batchnorm/ReadVariableOp_1¢@sequential_71/batch_normalization_647/batchnorm/ReadVariableOp_2¢Bsequential_71/batch_normalization_647/batchnorm/mul/ReadVariableOp¢>sequential_71/batch_normalization_648/batchnorm/ReadVariableOp¢@sequential_71/batch_normalization_648/batchnorm/ReadVariableOp_1¢@sequential_71/batch_normalization_648/batchnorm/ReadVariableOp_2¢Bsequential_71/batch_normalization_648/batchnorm/mul/ReadVariableOp¢.sequential_71/dense_711/BiasAdd/ReadVariableOp¢-sequential_71/dense_711/MatMul/ReadVariableOp¢.sequential_71/dense_712/BiasAdd/ReadVariableOp¢-sequential_71/dense_712/MatMul/ReadVariableOp¢.sequential_71/dense_713/BiasAdd/ReadVariableOp¢-sequential_71/dense_713/MatMul/ReadVariableOp¢.sequential_71/dense_714/BiasAdd/ReadVariableOp¢-sequential_71/dense_714/MatMul/ReadVariableOp¢.sequential_71/dense_715/BiasAdd/ReadVariableOp¢-sequential_71/dense_715/MatMul/ReadVariableOp¢.sequential_71/dense_716/BiasAdd/ReadVariableOp¢-sequential_71/dense_716/MatMul/ReadVariableOp¢.sequential_71/dense_717/BiasAdd/ReadVariableOp¢-sequential_71/dense_717/MatMul/ReadVariableOp¢.sequential_71/dense_718/BiasAdd/ReadVariableOp¢-sequential_71/dense_718/MatMul/ReadVariableOp¢.sequential_71/dense_719/BiasAdd/ReadVariableOp¢-sequential_71/dense_719/MatMul/ReadVariableOp¢.sequential_71/dense_720/BiasAdd/ReadVariableOp¢-sequential_71/dense_720/MatMul/ReadVariableOp
"sequential_71/normalization_71/subSubnormalization_71_input$sequential_71_normalization_71_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
#sequential_71/normalization_71/SqrtSqrt%sequential_71_normalization_71_sqrt_x*
T0*
_output_shapes

:m
(sequential_71/normalization_71/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¶
&sequential_71/normalization_71/MaximumMaximum'sequential_71/normalization_71/Sqrt:y:01sequential_71/normalization_71/Maximum/y:output:0*
T0*
_output_shapes

:·
&sequential_71/normalization_71/truedivRealDiv&sequential_71/normalization_71/sub:z:0*sequential_71/normalization_71/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_71/dense_711/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_711_matmul_readvariableop_resource*
_output_shapes

:i*
dtype0½
sequential_71/dense_711/MatMulMatMul*sequential_71/normalization_71/truediv:z:05sequential_71/dense_711/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi¢
.sequential_71/dense_711/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_711_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0¾
sequential_71/dense_711/BiasAddBiasAdd(sequential_71/dense_711/MatMul:product:06sequential_71/dense_711/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÂ
>sequential_71/batch_normalization_640/batchnorm/ReadVariableOpReadVariableOpGsequential_71_batch_normalization_640_batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0z
5sequential_71/batch_normalization_640/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_71/batch_normalization_640/batchnorm/addAddV2Fsequential_71/batch_normalization_640/batchnorm/ReadVariableOp:value:0>sequential_71/batch_normalization_640/batchnorm/add/y:output:0*
T0*
_output_shapes
:i
5sequential_71/batch_normalization_640/batchnorm/RsqrtRsqrt7sequential_71/batch_normalization_640/batchnorm/add:z:0*
T0*
_output_shapes
:iÊ
Bsequential_71/batch_normalization_640/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_71_batch_normalization_640_batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0æ
3sequential_71/batch_normalization_640/batchnorm/mulMul9sequential_71/batch_normalization_640/batchnorm/Rsqrt:y:0Jsequential_71/batch_normalization_640/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:iÑ
5sequential_71/batch_normalization_640/batchnorm/mul_1Mul(sequential_71/dense_711/BiasAdd:output:07sequential_71/batch_normalization_640/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÆ
@sequential_71/batch_normalization_640/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_71_batch_normalization_640_batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0ä
5sequential_71/batch_normalization_640/batchnorm/mul_2MulHsequential_71/batch_normalization_640/batchnorm/ReadVariableOp_1:value:07sequential_71/batch_normalization_640/batchnorm/mul:z:0*
T0*
_output_shapes
:iÆ
@sequential_71/batch_normalization_640/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_71_batch_normalization_640_batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0ä
3sequential_71/batch_normalization_640/batchnorm/subSubHsequential_71/batch_normalization_640/batchnorm/ReadVariableOp_2:value:09sequential_71/batch_normalization_640/batchnorm/mul_2:z:0*
T0*
_output_shapes
:iä
5sequential_71/batch_normalization_640/batchnorm/add_1AddV29sequential_71/batch_normalization_640/batchnorm/mul_1:z:07sequential_71/batch_normalization_640/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi¨
'sequential_71/leaky_re_lu_640/LeakyRelu	LeakyRelu9sequential_71/batch_normalization_640/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>¤
-sequential_71/dense_712/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_712_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0È
sequential_71/dense_712/MatMulMatMul5sequential_71/leaky_re_lu_640/LeakyRelu:activations:05sequential_71/dense_712/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi¢
.sequential_71/dense_712/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_712_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0¾
sequential_71/dense_712/BiasAddBiasAdd(sequential_71/dense_712/MatMul:product:06sequential_71/dense_712/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÂ
>sequential_71/batch_normalization_641/batchnorm/ReadVariableOpReadVariableOpGsequential_71_batch_normalization_641_batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0z
5sequential_71/batch_normalization_641/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_71/batch_normalization_641/batchnorm/addAddV2Fsequential_71/batch_normalization_641/batchnorm/ReadVariableOp:value:0>sequential_71/batch_normalization_641/batchnorm/add/y:output:0*
T0*
_output_shapes
:i
5sequential_71/batch_normalization_641/batchnorm/RsqrtRsqrt7sequential_71/batch_normalization_641/batchnorm/add:z:0*
T0*
_output_shapes
:iÊ
Bsequential_71/batch_normalization_641/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_71_batch_normalization_641_batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0æ
3sequential_71/batch_normalization_641/batchnorm/mulMul9sequential_71/batch_normalization_641/batchnorm/Rsqrt:y:0Jsequential_71/batch_normalization_641/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:iÑ
5sequential_71/batch_normalization_641/batchnorm/mul_1Mul(sequential_71/dense_712/BiasAdd:output:07sequential_71/batch_normalization_641/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÆ
@sequential_71/batch_normalization_641/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_71_batch_normalization_641_batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0ä
5sequential_71/batch_normalization_641/batchnorm/mul_2MulHsequential_71/batch_normalization_641/batchnorm/ReadVariableOp_1:value:07sequential_71/batch_normalization_641/batchnorm/mul:z:0*
T0*
_output_shapes
:iÆ
@sequential_71/batch_normalization_641/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_71_batch_normalization_641_batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0ä
3sequential_71/batch_normalization_641/batchnorm/subSubHsequential_71/batch_normalization_641/batchnorm/ReadVariableOp_2:value:09sequential_71/batch_normalization_641/batchnorm/mul_2:z:0*
T0*
_output_shapes
:iä
5sequential_71/batch_normalization_641/batchnorm/add_1AddV29sequential_71/batch_normalization_641/batchnorm/mul_1:z:07sequential_71/batch_normalization_641/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi¨
'sequential_71/leaky_re_lu_641/LeakyRelu	LeakyRelu9sequential_71/batch_normalization_641/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>¤
-sequential_71/dense_713/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_713_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0È
sequential_71/dense_713/MatMulMatMul5sequential_71/leaky_re_lu_641/LeakyRelu:activations:05sequential_71/dense_713/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi¢
.sequential_71/dense_713/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_713_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0¾
sequential_71/dense_713/BiasAddBiasAdd(sequential_71/dense_713/MatMul:product:06sequential_71/dense_713/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÂ
>sequential_71/batch_normalization_642/batchnorm/ReadVariableOpReadVariableOpGsequential_71_batch_normalization_642_batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0z
5sequential_71/batch_normalization_642/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_71/batch_normalization_642/batchnorm/addAddV2Fsequential_71/batch_normalization_642/batchnorm/ReadVariableOp:value:0>sequential_71/batch_normalization_642/batchnorm/add/y:output:0*
T0*
_output_shapes
:i
5sequential_71/batch_normalization_642/batchnorm/RsqrtRsqrt7sequential_71/batch_normalization_642/batchnorm/add:z:0*
T0*
_output_shapes
:iÊ
Bsequential_71/batch_normalization_642/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_71_batch_normalization_642_batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0æ
3sequential_71/batch_normalization_642/batchnorm/mulMul9sequential_71/batch_normalization_642/batchnorm/Rsqrt:y:0Jsequential_71/batch_normalization_642/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:iÑ
5sequential_71/batch_normalization_642/batchnorm/mul_1Mul(sequential_71/dense_713/BiasAdd:output:07sequential_71/batch_normalization_642/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÆ
@sequential_71/batch_normalization_642/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_71_batch_normalization_642_batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0ä
5sequential_71/batch_normalization_642/batchnorm/mul_2MulHsequential_71/batch_normalization_642/batchnorm/ReadVariableOp_1:value:07sequential_71/batch_normalization_642/batchnorm/mul:z:0*
T0*
_output_shapes
:iÆ
@sequential_71/batch_normalization_642/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_71_batch_normalization_642_batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0ä
3sequential_71/batch_normalization_642/batchnorm/subSubHsequential_71/batch_normalization_642/batchnorm/ReadVariableOp_2:value:09sequential_71/batch_normalization_642/batchnorm/mul_2:z:0*
T0*
_output_shapes
:iä
5sequential_71/batch_normalization_642/batchnorm/add_1AddV29sequential_71/batch_normalization_642/batchnorm/mul_1:z:07sequential_71/batch_normalization_642/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi¨
'sequential_71/leaky_re_lu_642/LeakyRelu	LeakyRelu9sequential_71/batch_normalization_642/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>¤
-sequential_71/dense_714/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_714_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0È
sequential_71/dense_714/MatMulMatMul5sequential_71/leaky_re_lu_642/LeakyRelu:activations:05sequential_71/dense_714/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi¢
.sequential_71/dense_714/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_714_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0¾
sequential_71/dense_714/BiasAddBiasAdd(sequential_71/dense_714/MatMul:product:06sequential_71/dense_714/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÂ
>sequential_71/batch_normalization_643/batchnorm/ReadVariableOpReadVariableOpGsequential_71_batch_normalization_643_batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0z
5sequential_71/batch_normalization_643/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_71/batch_normalization_643/batchnorm/addAddV2Fsequential_71/batch_normalization_643/batchnorm/ReadVariableOp:value:0>sequential_71/batch_normalization_643/batchnorm/add/y:output:0*
T0*
_output_shapes
:i
5sequential_71/batch_normalization_643/batchnorm/RsqrtRsqrt7sequential_71/batch_normalization_643/batchnorm/add:z:0*
T0*
_output_shapes
:iÊ
Bsequential_71/batch_normalization_643/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_71_batch_normalization_643_batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0æ
3sequential_71/batch_normalization_643/batchnorm/mulMul9sequential_71/batch_normalization_643/batchnorm/Rsqrt:y:0Jsequential_71/batch_normalization_643/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:iÑ
5sequential_71/batch_normalization_643/batchnorm/mul_1Mul(sequential_71/dense_714/BiasAdd:output:07sequential_71/batch_normalization_643/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÆ
@sequential_71/batch_normalization_643/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_71_batch_normalization_643_batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0ä
5sequential_71/batch_normalization_643/batchnorm/mul_2MulHsequential_71/batch_normalization_643/batchnorm/ReadVariableOp_1:value:07sequential_71/batch_normalization_643/batchnorm/mul:z:0*
T0*
_output_shapes
:iÆ
@sequential_71/batch_normalization_643/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_71_batch_normalization_643_batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0ä
3sequential_71/batch_normalization_643/batchnorm/subSubHsequential_71/batch_normalization_643/batchnorm/ReadVariableOp_2:value:09sequential_71/batch_normalization_643/batchnorm/mul_2:z:0*
T0*
_output_shapes
:iä
5sequential_71/batch_normalization_643/batchnorm/add_1AddV29sequential_71/batch_normalization_643/batchnorm/mul_1:z:07sequential_71/batch_normalization_643/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi¨
'sequential_71/leaky_re_lu_643/LeakyRelu	LeakyRelu9sequential_71/batch_normalization_643/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>¤
-sequential_71/dense_715/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_715_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0È
sequential_71/dense_715/MatMulMatMul5sequential_71/leaky_re_lu_643/LeakyRelu:activations:05sequential_71/dense_715/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi¢
.sequential_71/dense_715/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_715_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0¾
sequential_71/dense_715/BiasAddBiasAdd(sequential_71/dense_715/MatMul:product:06sequential_71/dense_715/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÂ
>sequential_71/batch_normalization_644/batchnorm/ReadVariableOpReadVariableOpGsequential_71_batch_normalization_644_batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0z
5sequential_71/batch_normalization_644/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_71/batch_normalization_644/batchnorm/addAddV2Fsequential_71/batch_normalization_644/batchnorm/ReadVariableOp:value:0>sequential_71/batch_normalization_644/batchnorm/add/y:output:0*
T0*
_output_shapes
:i
5sequential_71/batch_normalization_644/batchnorm/RsqrtRsqrt7sequential_71/batch_normalization_644/batchnorm/add:z:0*
T0*
_output_shapes
:iÊ
Bsequential_71/batch_normalization_644/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_71_batch_normalization_644_batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0æ
3sequential_71/batch_normalization_644/batchnorm/mulMul9sequential_71/batch_normalization_644/batchnorm/Rsqrt:y:0Jsequential_71/batch_normalization_644/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:iÑ
5sequential_71/batch_normalization_644/batchnorm/mul_1Mul(sequential_71/dense_715/BiasAdd:output:07sequential_71/batch_normalization_644/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÆ
@sequential_71/batch_normalization_644/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_71_batch_normalization_644_batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0ä
5sequential_71/batch_normalization_644/batchnorm/mul_2MulHsequential_71/batch_normalization_644/batchnorm/ReadVariableOp_1:value:07sequential_71/batch_normalization_644/batchnorm/mul:z:0*
T0*
_output_shapes
:iÆ
@sequential_71/batch_normalization_644/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_71_batch_normalization_644_batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0ä
3sequential_71/batch_normalization_644/batchnorm/subSubHsequential_71/batch_normalization_644/batchnorm/ReadVariableOp_2:value:09sequential_71/batch_normalization_644/batchnorm/mul_2:z:0*
T0*
_output_shapes
:iä
5sequential_71/batch_normalization_644/batchnorm/add_1AddV29sequential_71/batch_normalization_644/batchnorm/mul_1:z:07sequential_71/batch_normalization_644/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi¨
'sequential_71/leaky_re_lu_644/LeakyRelu	LeakyRelu9sequential_71/batch_normalization_644/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>¤
-sequential_71/dense_716/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_716_matmul_readvariableop_resource*
_output_shapes

:i=*
dtype0È
sequential_71/dense_716/MatMulMatMul5sequential_71/leaky_re_lu_644/LeakyRelu:activations:05sequential_71/dense_716/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=¢
.sequential_71/dense_716/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_716_biasadd_readvariableop_resource*
_output_shapes
:=*
dtype0¾
sequential_71/dense_716/BiasAddBiasAdd(sequential_71/dense_716/MatMul:product:06sequential_71/dense_716/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=Â
>sequential_71/batch_normalization_645/batchnorm/ReadVariableOpReadVariableOpGsequential_71_batch_normalization_645_batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0z
5sequential_71/batch_normalization_645/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_71/batch_normalization_645/batchnorm/addAddV2Fsequential_71/batch_normalization_645/batchnorm/ReadVariableOp:value:0>sequential_71/batch_normalization_645/batchnorm/add/y:output:0*
T0*
_output_shapes
:=
5sequential_71/batch_normalization_645/batchnorm/RsqrtRsqrt7sequential_71/batch_normalization_645/batchnorm/add:z:0*
T0*
_output_shapes
:=Ê
Bsequential_71/batch_normalization_645/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_71_batch_normalization_645_batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0æ
3sequential_71/batch_normalization_645/batchnorm/mulMul9sequential_71/batch_normalization_645/batchnorm/Rsqrt:y:0Jsequential_71/batch_normalization_645/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=Ñ
5sequential_71/batch_normalization_645/batchnorm/mul_1Mul(sequential_71/dense_716/BiasAdd:output:07sequential_71/batch_normalization_645/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=Æ
@sequential_71/batch_normalization_645/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_71_batch_normalization_645_batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0ä
5sequential_71/batch_normalization_645/batchnorm/mul_2MulHsequential_71/batch_normalization_645/batchnorm/ReadVariableOp_1:value:07sequential_71/batch_normalization_645/batchnorm/mul:z:0*
T0*
_output_shapes
:=Æ
@sequential_71/batch_normalization_645/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_71_batch_normalization_645_batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0ä
3sequential_71/batch_normalization_645/batchnorm/subSubHsequential_71/batch_normalization_645/batchnorm/ReadVariableOp_2:value:09sequential_71/batch_normalization_645/batchnorm/mul_2:z:0*
T0*
_output_shapes
:=ä
5sequential_71/batch_normalization_645/batchnorm/add_1AddV29sequential_71/batch_normalization_645/batchnorm/mul_1:z:07sequential_71/batch_normalization_645/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=¨
'sequential_71/leaky_re_lu_645/LeakyRelu	LeakyRelu9sequential_71/batch_normalization_645/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>¤
-sequential_71/dense_717/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_717_matmul_readvariableop_resource*
_output_shapes

:==*
dtype0È
sequential_71/dense_717/MatMulMatMul5sequential_71/leaky_re_lu_645/LeakyRelu:activations:05sequential_71/dense_717/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=¢
.sequential_71/dense_717/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_717_biasadd_readvariableop_resource*
_output_shapes
:=*
dtype0¾
sequential_71/dense_717/BiasAddBiasAdd(sequential_71/dense_717/MatMul:product:06sequential_71/dense_717/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=Â
>sequential_71/batch_normalization_646/batchnorm/ReadVariableOpReadVariableOpGsequential_71_batch_normalization_646_batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0z
5sequential_71/batch_normalization_646/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_71/batch_normalization_646/batchnorm/addAddV2Fsequential_71/batch_normalization_646/batchnorm/ReadVariableOp:value:0>sequential_71/batch_normalization_646/batchnorm/add/y:output:0*
T0*
_output_shapes
:=
5sequential_71/batch_normalization_646/batchnorm/RsqrtRsqrt7sequential_71/batch_normalization_646/batchnorm/add:z:0*
T0*
_output_shapes
:=Ê
Bsequential_71/batch_normalization_646/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_71_batch_normalization_646_batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0æ
3sequential_71/batch_normalization_646/batchnorm/mulMul9sequential_71/batch_normalization_646/batchnorm/Rsqrt:y:0Jsequential_71/batch_normalization_646/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=Ñ
5sequential_71/batch_normalization_646/batchnorm/mul_1Mul(sequential_71/dense_717/BiasAdd:output:07sequential_71/batch_normalization_646/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=Æ
@sequential_71/batch_normalization_646/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_71_batch_normalization_646_batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0ä
5sequential_71/batch_normalization_646/batchnorm/mul_2MulHsequential_71/batch_normalization_646/batchnorm/ReadVariableOp_1:value:07sequential_71/batch_normalization_646/batchnorm/mul:z:0*
T0*
_output_shapes
:=Æ
@sequential_71/batch_normalization_646/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_71_batch_normalization_646_batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0ä
3sequential_71/batch_normalization_646/batchnorm/subSubHsequential_71/batch_normalization_646/batchnorm/ReadVariableOp_2:value:09sequential_71/batch_normalization_646/batchnorm/mul_2:z:0*
T0*
_output_shapes
:=ä
5sequential_71/batch_normalization_646/batchnorm/add_1AddV29sequential_71/batch_normalization_646/batchnorm/mul_1:z:07sequential_71/batch_normalization_646/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=¨
'sequential_71/leaky_re_lu_646/LeakyRelu	LeakyRelu9sequential_71/batch_normalization_646/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>¤
-sequential_71/dense_718/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_718_matmul_readvariableop_resource*
_output_shapes

:=7*
dtype0È
sequential_71/dense_718/MatMulMatMul5sequential_71/leaky_re_lu_646/LeakyRelu:activations:05sequential_71/dense_718/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¢
.sequential_71/dense_718/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_718_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0¾
sequential_71/dense_718/BiasAddBiasAdd(sequential_71/dense_718/MatMul:product:06sequential_71/dense_718/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7Â
>sequential_71/batch_normalization_647/batchnorm/ReadVariableOpReadVariableOpGsequential_71_batch_normalization_647_batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0z
5sequential_71/batch_normalization_647/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_71/batch_normalization_647/batchnorm/addAddV2Fsequential_71/batch_normalization_647/batchnorm/ReadVariableOp:value:0>sequential_71/batch_normalization_647/batchnorm/add/y:output:0*
T0*
_output_shapes
:7
5sequential_71/batch_normalization_647/batchnorm/RsqrtRsqrt7sequential_71/batch_normalization_647/batchnorm/add:z:0*
T0*
_output_shapes
:7Ê
Bsequential_71/batch_normalization_647/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_71_batch_normalization_647_batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0æ
3sequential_71/batch_normalization_647/batchnorm/mulMul9sequential_71/batch_normalization_647/batchnorm/Rsqrt:y:0Jsequential_71/batch_normalization_647/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7Ñ
5sequential_71/batch_normalization_647/batchnorm/mul_1Mul(sequential_71/dense_718/BiasAdd:output:07sequential_71/batch_normalization_647/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7Æ
@sequential_71/batch_normalization_647/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_71_batch_normalization_647_batchnorm_readvariableop_1_resource*
_output_shapes
:7*
dtype0ä
5sequential_71/batch_normalization_647/batchnorm/mul_2MulHsequential_71/batch_normalization_647/batchnorm/ReadVariableOp_1:value:07sequential_71/batch_normalization_647/batchnorm/mul:z:0*
T0*
_output_shapes
:7Æ
@sequential_71/batch_normalization_647/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_71_batch_normalization_647_batchnorm_readvariableop_2_resource*
_output_shapes
:7*
dtype0ä
3sequential_71/batch_normalization_647/batchnorm/subSubHsequential_71/batch_normalization_647/batchnorm/ReadVariableOp_2:value:09sequential_71/batch_normalization_647/batchnorm/mul_2:z:0*
T0*
_output_shapes
:7ä
5sequential_71/batch_normalization_647/batchnorm/add_1AddV29sequential_71/batch_normalization_647/batchnorm/mul_1:z:07sequential_71/batch_normalization_647/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¨
'sequential_71/leaky_re_lu_647/LeakyRelu	LeakyRelu9sequential_71/batch_normalization_647/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>¤
-sequential_71/dense_719/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_719_matmul_readvariableop_resource*
_output_shapes

:77*
dtype0È
sequential_71/dense_719/MatMulMatMul5sequential_71/leaky_re_lu_647/LeakyRelu:activations:05sequential_71/dense_719/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¢
.sequential_71/dense_719/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_719_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0¾
sequential_71/dense_719/BiasAddBiasAdd(sequential_71/dense_719/MatMul:product:06sequential_71/dense_719/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7Â
>sequential_71/batch_normalization_648/batchnorm/ReadVariableOpReadVariableOpGsequential_71_batch_normalization_648_batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0z
5sequential_71/batch_normalization_648/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_71/batch_normalization_648/batchnorm/addAddV2Fsequential_71/batch_normalization_648/batchnorm/ReadVariableOp:value:0>sequential_71/batch_normalization_648/batchnorm/add/y:output:0*
T0*
_output_shapes
:7
5sequential_71/batch_normalization_648/batchnorm/RsqrtRsqrt7sequential_71/batch_normalization_648/batchnorm/add:z:0*
T0*
_output_shapes
:7Ê
Bsequential_71/batch_normalization_648/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_71_batch_normalization_648_batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0æ
3sequential_71/batch_normalization_648/batchnorm/mulMul9sequential_71/batch_normalization_648/batchnorm/Rsqrt:y:0Jsequential_71/batch_normalization_648/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7Ñ
5sequential_71/batch_normalization_648/batchnorm/mul_1Mul(sequential_71/dense_719/BiasAdd:output:07sequential_71/batch_normalization_648/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7Æ
@sequential_71/batch_normalization_648/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_71_batch_normalization_648_batchnorm_readvariableop_1_resource*
_output_shapes
:7*
dtype0ä
5sequential_71/batch_normalization_648/batchnorm/mul_2MulHsequential_71/batch_normalization_648/batchnorm/ReadVariableOp_1:value:07sequential_71/batch_normalization_648/batchnorm/mul:z:0*
T0*
_output_shapes
:7Æ
@sequential_71/batch_normalization_648/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_71_batch_normalization_648_batchnorm_readvariableop_2_resource*
_output_shapes
:7*
dtype0ä
3sequential_71/batch_normalization_648/batchnorm/subSubHsequential_71/batch_normalization_648/batchnorm/ReadVariableOp_2:value:09sequential_71/batch_normalization_648/batchnorm/mul_2:z:0*
T0*
_output_shapes
:7ä
5sequential_71/batch_normalization_648/batchnorm/add_1AddV29sequential_71/batch_normalization_648/batchnorm/mul_1:z:07sequential_71/batch_normalization_648/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¨
'sequential_71/leaky_re_lu_648/LeakyRelu	LeakyRelu9sequential_71/batch_normalization_648/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>¤
-sequential_71/dense_720/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_720_matmul_readvariableop_resource*
_output_shapes

:7*
dtype0È
sequential_71/dense_720/MatMulMatMul5sequential_71/leaky_re_lu_648/LeakyRelu:activations:05sequential_71/dense_720/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_71/dense_720/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_720_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_71/dense_720/BiasAddBiasAdd(sequential_71/dense_720/MatMul:product:06sequential_71/dense_720/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_71/dense_720/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿü
NoOpNoOp?^sequential_71/batch_normalization_640/batchnorm/ReadVariableOpA^sequential_71/batch_normalization_640/batchnorm/ReadVariableOp_1A^sequential_71/batch_normalization_640/batchnorm/ReadVariableOp_2C^sequential_71/batch_normalization_640/batchnorm/mul/ReadVariableOp?^sequential_71/batch_normalization_641/batchnorm/ReadVariableOpA^sequential_71/batch_normalization_641/batchnorm/ReadVariableOp_1A^sequential_71/batch_normalization_641/batchnorm/ReadVariableOp_2C^sequential_71/batch_normalization_641/batchnorm/mul/ReadVariableOp?^sequential_71/batch_normalization_642/batchnorm/ReadVariableOpA^sequential_71/batch_normalization_642/batchnorm/ReadVariableOp_1A^sequential_71/batch_normalization_642/batchnorm/ReadVariableOp_2C^sequential_71/batch_normalization_642/batchnorm/mul/ReadVariableOp?^sequential_71/batch_normalization_643/batchnorm/ReadVariableOpA^sequential_71/batch_normalization_643/batchnorm/ReadVariableOp_1A^sequential_71/batch_normalization_643/batchnorm/ReadVariableOp_2C^sequential_71/batch_normalization_643/batchnorm/mul/ReadVariableOp?^sequential_71/batch_normalization_644/batchnorm/ReadVariableOpA^sequential_71/batch_normalization_644/batchnorm/ReadVariableOp_1A^sequential_71/batch_normalization_644/batchnorm/ReadVariableOp_2C^sequential_71/batch_normalization_644/batchnorm/mul/ReadVariableOp?^sequential_71/batch_normalization_645/batchnorm/ReadVariableOpA^sequential_71/batch_normalization_645/batchnorm/ReadVariableOp_1A^sequential_71/batch_normalization_645/batchnorm/ReadVariableOp_2C^sequential_71/batch_normalization_645/batchnorm/mul/ReadVariableOp?^sequential_71/batch_normalization_646/batchnorm/ReadVariableOpA^sequential_71/batch_normalization_646/batchnorm/ReadVariableOp_1A^sequential_71/batch_normalization_646/batchnorm/ReadVariableOp_2C^sequential_71/batch_normalization_646/batchnorm/mul/ReadVariableOp?^sequential_71/batch_normalization_647/batchnorm/ReadVariableOpA^sequential_71/batch_normalization_647/batchnorm/ReadVariableOp_1A^sequential_71/batch_normalization_647/batchnorm/ReadVariableOp_2C^sequential_71/batch_normalization_647/batchnorm/mul/ReadVariableOp?^sequential_71/batch_normalization_648/batchnorm/ReadVariableOpA^sequential_71/batch_normalization_648/batchnorm/ReadVariableOp_1A^sequential_71/batch_normalization_648/batchnorm/ReadVariableOp_2C^sequential_71/batch_normalization_648/batchnorm/mul/ReadVariableOp/^sequential_71/dense_711/BiasAdd/ReadVariableOp.^sequential_71/dense_711/MatMul/ReadVariableOp/^sequential_71/dense_712/BiasAdd/ReadVariableOp.^sequential_71/dense_712/MatMul/ReadVariableOp/^sequential_71/dense_713/BiasAdd/ReadVariableOp.^sequential_71/dense_713/MatMul/ReadVariableOp/^sequential_71/dense_714/BiasAdd/ReadVariableOp.^sequential_71/dense_714/MatMul/ReadVariableOp/^sequential_71/dense_715/BiasAdd/ReadVariableOp.^sequential_71/dense_715/MatMul/ReadVariableOp/^sequential_71/dense_716/BiasAdd/ReadVariableOp.^sequential_71/dense_716/MatMul/ReadVariableOp/^sequential_71/dense_717/BiasAdd/ReadVariableOp.^sequential_71/dense_717/MatMul/ReadVariableOp/^sequential_71/dense_718/BiasAdd/ReadVariableOp.^sequential_71/dense_718/MatMul/ReadVariableOp/^sequential_71/dense_719/BiasAdd/ReadVariableOp.^sequential_71/dense_719/MatMul/ReadVariableOp/^sequential_71/dense_720/BiasAdd/ReadVariableOp.^sequential_71/dense_720/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
>sequential_71/batch_normalization_640/batchnorm/ReadVariableOp>sequential_71/batch_normalization_640/batchnorm/ReadVariableOp2
@sequential_71/batch_normalization_640/batchnorm/ReadVariableOp_1@sequential_71/batch_normalization_640/batchnorm/ReadVariableOp_12
@sequential_71/batch_normalization_640/batchnorm/ReadVariableOp_2@sequential_71/batch_normalization_640/batchnorm/ReadVariableOp_22
Bsequential_71/batch_normalization_640/batchnorm/mul/ReadVariableOpBsequential_71/batch_normalization_640/batchnorm/mul/ReadVariableOp2
>sequential_71/batch_normalization_641/batchnorm/ReadVariableOp>sequential_71/batch_normalization_641/batchnorm/ReadVariableOp2
@sequential_71/batch_normalization_641/batchnorm/ReadVariableOp_1@sequential_71/batch_normalization_641/batchnorm/ReadVariableOp_12
@sequential_71/batch_normalization_641/batchnorm/ReadVariableOp_2@sequential_71/batch_normalization_641/batchnorm/ReadVariableOp_22
Bsequential_71/batch_normalization_641/batchnorm/mul/ReadVariableOpBsequential_71/batch_normalization_641/batchnorm/mul/ReadVariableOp2
>sequential_71/batch_normalization_642/batchnorm/ReadVariableOp>sequential_71/batch_normalization_642/batchnorm/ReadVariableOp2
@sequential_71/batch_normalization_642/batchnorm/ReadVariableOp_1@sequential_71/batch_normalization_642/batchnorm/ReadVariableOp_12
@sequential_71/batch_normalization_642/batchnorm/ReadVariableOp_2@sequential_71/batch_normalization_642/batchnorm/ReadVariableOp_22
Bsequential_71/batch_normalization_642/batchnorm/mul/ReadVariableOpBsequential_71/batch_normalization_642/batchnorm/mul/ReadVariableOp2
>sequential_71/batch_normalization_643/batchnorm/ReadVariableOp>sequential_71/batch_normalization_643/batchnorm/ReadVariableOp2
@sequential_71/batch_normalization_643/batchnorm/ReadVariableOp_1@sequential_71/batch_normalization_643/batchnorm/ReadVariableOp_12
@sequential_71/batch_normalization_643/batchnorm/ReadVariableOp_2@sequential_71/batch_normalization_643/batchnorm/ReadVariableOp_22
Bsequential_71/batch_normalization_643/batchnorm/mul/ReadVariableOpBsequential_71/batch_normalization_643/batchnorm/mul/ReadVariableOp2
>sequential_71/batch_normalization_644/batchnorm/ReadVariableOp>sequential_71/batch_normalization_644/batchnorm/ReadVariableOp2
@sequential_71/batch_normalization_644/batchnorm/ReadVariableOp_1@sequential_71/batch_normalization_644/batchnorm/ReadVariableOp_12
@sequential_71/batch_normalization_644/batchnorm/ReadVariableOp_2@sequential_71/batch_normalization_644/batchnorm/ReadVariableOp_22
Bsequential_71/batch_normalization_644/batchnorm/mul/ReadVariableOpBsequential_71/batch_normalization_644/batchnorm/mul/ReadVariableOp2
>sequential_71/batch_normalization_645/batchnorm/ReadVariableOp>sequential_71/batch_normalization_645/batchnorm/ReadVariableOp2
@sequential_71/batch_normalization_645/batchnorm/ReadVariableOp_1@sequential_71/batch_normalization_645/batchnorm/ReadVariableOp_12
@sequential_71/batch_normalization_645/batchnorm/ReadVariableOp_2@sequential_71/batch_normalization_645/batchnorm/ReadVariableOp_22
Bsequential_71/batch_normalization_645/batchnorm/mul/ReadVariableOpBsequential_71/batch_normalization_645/batchnorm/mul/ReadVariableOp2
>sequential_71/batch_normalization_646/batchnorm/ReadVariableOp>sequential_71/batch_normalization_646/batchnorm/ReadVariableOp2
@sequential_71/batch_normalization_646/batchnorm/ReadVariableOp_1@sequential_71/batch_normalization_646/batchnorm/ReadVariableOp_12
@sequential_71/batch_normalization_646/batchnorm/ReadVariableOp_2@sequential_71/batch_normalization_646/batchnorm/ReadVariableOp_22
Bsequential_71/batch_normalization_646/batchnorm/mul/ReadVariableOpBsequential_71/batch_normalization_646/batchnorm/mul/ReadVariableOp2
>sequential_71/batch_normalization_647/batchnorm/ReadVariableOp>sequential_71/batch_normalization_647/batchnorm/ReadVariableOp2
@sequential_71/batch_normalization_647/batchnorm/ReadVariableOp_1@sequential_71/batch_normalization_647/batchnorm/ReadVariableOp_12
@sequential_71/batch_normalization_647/batchnorm/ReadVariableOp_2@sequential_71/batch_normalization_647/batchnorm/ReadVariableOp_22
Bsequential_71/batch_normalization_647/batchnorm/mul/ReadVariableOpBsequential_71/batch_normalization_647/batchnorm/mul/ReadVariableOp2
>sequential_71/batch_normalization_648/batchnorm/ReadVariableOp>sequential_71/batch_normalization_648/batchnorm/ReadVariableOp2
@sequential_71/batch_normalization_648/batchnorm/ReadVariableOp_1@sequential_71/batch_normalization_648/batchnorm/ReadVariableOp_12
@sequential_71/batch_normalization_648/batchnorm/ReadVariableOp_2@sequential_71/batch_normalization_648/batchnorm/ReadVariableOp_22
Bsequential_71/batch_normalization_648/batchnorm/mul/ReadVariableOpBsequential_71/batch_normalization_648/batchnorm/mul/ReadVariableOp2`
.sequential_71/dense_711/BiasAdd/ReadVariableOp.sequential_71/dense_711/BiasAdd/ReadVariableOp2^
-sequential_71/dense_711/MatMul/ReadVariableOp-sequential_71/dense_711/MatMul/ReadVariableOp2`
.sequential_71/dense_712/BiasAdd/ReadVariableOp.sequential_71/dense_712/BiasAdd/ReadVariableOp2^
-sequential_71/dense_712/MatMul/ReadVariableOp-sequential_71/dense_712/MatMul/ReadVariableOp2`
.sequential_71/dense_713/BiasAdd/ReadVariableOp.sequential_71/dense_713/BiasAdd/ReadVariableOp2^
-sequential_71/dense_713/MatMul/ReadVariableOp-sequential_71/dense_713/MatMul/ReadVariableOp2`
.sequential_71/dense_714/BiasAdd/ReadVariableOp.sequential_71/dense_714/BiasAdd/ReadVariableOp2^
-sequential_71/dense_714/MatMul/ReadVariableOp-sequential_71/dense_714/MatMul/ReadVariableOp2`
.sequential_71/dense_715/BiasAdd/ReadVariableOp.sequential_71/dense_715/BiasAdd/ReadVariableOp2^
-sequential_71/dense_715/MatMul/ReadVariableOp-sequential_71/dense_715/MatMul/ReadVariableOp2`
.sequential_71/dense_716/BiasAdd/ReadVariableOp.sequential_71/dense_716/BiasAdd/ReadVariableOp2^
-sequential_71/dense_716/MatMul/ReadVariableOp-sequential_71/dense_716/MatMul/ReadVariableOp2`
.sequential_71/dense_717/BiasAdd/ReadVariableOp.sequential_71/dense_717/BiasAdd/ReadVariableOp2^
-sequential_71/dense_717/MatMul/ReadVariableOp-sequential_71/dense_717/MatMul/ReadVariableOp2`
.sequential_71/dense_718/BiasAdd/ReadVariableOp.sequential_71/dense_718/BiasAdd/ReadVariableOp2^
-sequential_71/dense_718/MatMul/ReadVariableOp-sequential_71/dense_718/MatMul/ReadVariableOp2`
.sequential_71/dense_719/BiasAdd/ReadVariableOp.sequential_71/dense_719/BiasAdd/ReadVariableOp2^
-sequential_71/dense_719/MatMul/ReadVariableOp-sequential_71/dense_719/MatMul/ReadVariableOp2`
.sequential_71/dense_720/BiasAdd/ReadVariableOp.sequential_71/dense_720/BiasAdd/ReadVariableOp2^
-sequential_71/dense_720/MatMul/ReadVariableOp-sequential_71/dense_720/MatMul/ReadVariableOp:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_71_input:$ 

_output_shapes

::$ 

_output_shapes

:
¥
Þ
F__inference_dense_712_layer_call_and_return_conditional_losses_1121585

inputs0
matmul_readvariableop_resource:ii-
biasadd_readvariableop_resource:i
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_712/kernel/Regularizer/Abs/ReadVariableOp¢2dense_712/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿir
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:i*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿig
"dense_712/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_712/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_712/kernel/Regularizer/AbsAbs7dense_712/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_712/kernel/Regularizer/SumSum$dense_712/kernel/Regularizer/Abs:y:0-dense_712/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_712/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_712/kernel/Regularizer/mulMul+dense_712/kernel/Regularizer/mul/x:output:0)dense_712/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_712/kernel/Regularizer/addAddV2+dense_712/kernel/Regularizer/Const:output:0$dense_712/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_712/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_712/kernel/Regularizer/SquareSquare:dense_712/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_712/kernel/Regularizer/Sum_1Sum'dense_712/kernel/Regularizer/Square:y:0-dense_712/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_712/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_712/kernel/Regularizer/mul_1Mul-dense_712/kernel/Regularizer/mul_1/x:output:0+dense_712/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_712/kernel/Regularizer/add_1AddV2$dense_712/kernel/Regularizer/add:z:0&dense_712/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÞ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_712/kernel/Regularizer/Abs/ReadVariableOp3^dense_712/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_712/kernel/Regularizer/Abs/ReadVariableOp/dense_712/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_712/kernel/Regularizer/Square/ReadVariableOp2dense_712/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
ûÆ
Ã!
J__inference_sequential_71_layer_call_and_return_conditional_losses_1122088

inputs
normalization_71_sub_y
normalization_71_sqrt_x#
dense_711_1121539:i
dense_711_1121541:i-
batch_normalization_640_1121544:i-
batch_normalization_640_1121546:i-
batch_normalization_640_1121548:i-
batch_normalization_640_1121550:i#
dense_712_1121586:ii
dense_712_1121588:i-
batch_normalization_641_1121591:i-
batch_normalization_641_1121593:i-
batch_normalization_641_1121595:i-
batch_normalization_641_1121597:i#
dense_713_1121633:ii
dense_713_1121635:i-
batch_normalization_642_1121638:i-
batch_normalization_642_1121640:i-
batch_normalization_642_1121642:i-
batch_normalization_642_1121644:i#
dense_714_1121680:ii
dense_714_1121682:i-
batch_normalization_643_1121685:i-
batch_normalization_643_1121687:i-
batch_normalization_643_1121689:i-
batch_normalization_643_1121691:i#
dense_715_1121727:ii
dense_715_1121729:i-
batch_normalization_644_1121732:i-
batch_normalization_644_1121734:i-
batch_normalization_644_1121736:i-
batch_normalization_644_1121738:i#
dense_716_1121774:i=
dense_716_1121776:=-
batch_normalization_645_1121779:=-
batch_normalization_645_1121781:=-
batch_normalization_645_1121783:=-
batch_normalization_645_1121785:=#
dense_717_1121821:==
dense_717_1121823:=-
batch_normalization_646_1121826:=-
batch_normalization_646_1121828:=-
batch_normalization_646_1121830:=-
batch_normalization_646_1121832:=#
dense_718_1121868:=7
dense_718_1121870:7-
batch_normalization_647_1121873:7-
batch_normalization_647_1121875:7-
batch_normalization_647_1121877:7-
batch_normalization_647_1121879:7#
dense_719_1121915:77
dense_719_1121917:7-
batch_normalization_648_1121920:7-
batch_normalization_648_1121922:7-
batch_normalization_648_1121924:7-
batch_normalization_648_1121926:7#
dense_720_1121947:7
dense_720_1121949:
identity¢/batch_normalization_640/StatefulPartitionedCall¢/batch_normalization_641/StatefulPartitionedCall¢/batch_normalization_642/StatefulPartitionedCall¢/batch_normalization_643/StatefulPartitionedCall¢/batch_normalization_644/StatefulPartitionedCall¢/batch_normalization_645/StatefulPartitionedCall¢/batch_normalization_646/StatefulPartitionedCall¢/batch_normalization_647/StatefulPartitionedCall¢/batch_normalization_648/StatefulPartitionedCall¢!dense_711/StatefulPartitionedCall¢/dense_711/kernel/Regularizer/Abs/ReadVariableOp¢2dense_711/kernel/Regularizer/Square/ReadVariableOp¢!dense_712/StatefulPartitionedCall¢/dense_712/kernel/Regularizer/Abs/ReadVariableOp¢2dense_712/kernel/Regularizer/Square/ReadVariableOp¢!dense_713/StatefulPartitionedCall¢/dense_713/kernel/Regularizer/Abs/ReadVariableOp¢2dense_713/kernel/Regularizer/Square/ReadVariableOp¢!dense_714/StatefulPartitionedCall¢/dense_714/kernel/Regularizer/Abs/ReadVariableOp¢2dense_714/kernel/Regularizer/Square/ReadVariableOp¢!dense_715/StatefulPartitionedCall¢/dense_715/kernel/Regularizer/Abs/ReadVariableOp¢2dense_715/kernel/Regularizer/Square/ReadVariableOp¢!dense_716/StatefulPartitionedCall¢/dense_716/kernel/Regularizer/Abs/ReadVariableOp¢2dense_716/kernel/Regularizer/Square/ReadVariableOp¢!dense_717/StatefulPartitionedCall¢/dense_717/kernel/Regularizer/Abs/ReadVariableOp¢2dense_717/kernel/Regularizer/Square/ReadVariableOp¢!dense_718/StatefulPartitionedCall¢/dense_718/kernel/Regularizer/Abs/ReadVariableOp¢2dense_718/kernel/Regularizer/Square/ReadVariableOp¢!dense_719/StatefulPartitionedCall¢/dense_719/kernel/Regularizer/Abs/ReadVariableOp¢2dense_719/kernel/Regularizer/Square/ReadVariableOp¢!dense_720/StatefulPartitionedCallm
normalization_71/subSubinputsnormalization_71_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_71/SqrtSqrtnormalization_71_sqrt_x*
T0*
_output_shapes

:_
normalization_71/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_71/MaximumMaximumnormalization_71/Sqrt:y:0#normalization_71/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_71/truedivRealDivnormalization_71/sub:z:0normalization_71/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_711/StatefulPartitionedCallStatefulPartitionedCallnormalization_71/truediv:z:0dense_711_1121539dense_711_1121541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_711_layer_call_and_return_conditional_losses_1121538
/batch_normalization_640/StatefulPartitionedCallStatefulPartitionedCall*dense_711/StatefulPartitionedCall:output:0batch_normalization_640_1121544batch_normalization_640_1121546batch_normalization_640_1121548batch_normalization_640_1121550*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_640_layer_call_and_return_conditional_losses_1120785ù
leaky_re_lu_640/PartitionedCallPartitionedCall8batch_normalization_640/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_640_layer_call_and_return_conditional_losses_1121558
!dense_712/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_640/PartitionedCall:output:0dense_712_1121586dense_712_1121588*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_712_layer_call_and_return_conditional_losses_1121585
/batch_normalization_641/StatefulPartitionedCallStatefulPartitionedCall*dense_712/StatefulPartitionedCall:output:0batch_normalization_641_1121591batch_normalization_641_1121593batch_normalization_641_1121595batch_normalization_641_1121597*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_641_layer_call_and_return_conditional_losses_1120867ù
leaky_re_lu_641/PartitionedCallPartitionedCall8batch_normalization_641/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_641_layer_call_and_return_conditional_losses_1121605
!dense_713/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_641/PartitionedCall:output:0dense_713_1121633dense_713_1121635*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_713_layer_call_and_return_conditional_losses_1121632
/batch_normalization_642/StatefulPartitionedCallStatefulPartitionedCall*dense_713/StatefulPartitionedCall:output:0batch_normalization_642_1121638batch_normalization_642_1121640batch_normalization_642_1121642batch_normalization_642_1121644*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_642_layer_call_and_return_conditional_losses_1120949ù
leaky_re_lu_642/PartitionedCallPartitionedCall8batch_normalization_642/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_642_layer_call_and_return_conditional_losses_1121652
!dense_714/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_642/PartitionedCall:output:0dense_714_1121680dense_714_1121682*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_714_layer_call_and_return_conditional_losses_1121679
/batch_normalization_643/StatefulPartitionedCallStatefulPartitionedCall*dense_714/StatefulPartitionedCall:output:0batch_normalization_643_1121685batch_normalization_643_1121687batch_normalization_643_1121689batch_normalization_643_1121691*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_643_layer_call_and_return_conditional_losses_1121031ù
leaky_re_lu_643/PartitionedCallPartitionedCall8batch_normalization_643/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_643_layer_call_and_return_conditional_losses_1121699
!dense_715/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_643/PartitionedCall:output:0dense_715_1121727dense_715_1121729*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_715_layer_call_and_return_conditional_losses_1121726
/batch_normalization_644/StatefulPartitionedCallStatefulPartitionedCall*dense_715/StatefulPartitionedCall:output:0batch_normalization_644_1121732batch_normalization_644_1121734batch_normalization_644_1121736batch_normalization_644_1121738*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_644_layer_call_and_return_conditional_losses_1121113ù
leaky_re_lu_644/PartitionedCallPartitionedCall8batch_normalization_644/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_644_layer_call_and_return_conditional_losses_1121746
!dense_716/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_644/PartitionedCall:output:0dense_716_1121774dense_716_1121776*
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
GPU 2J 8 *O
fJRH
F__inference_dense_716_layer_call_and_return_conditional_losses_1121773
/batch_normalization_645/StatefulPartitionedCallStatefulPartitionedCall*dense_716/StatefulPartitionedCall:output:0batch_normalization_645_1121779batch_normalization_645_1121781batch_normalization_645_1121783batch_normalization_645_1121785*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_645_layer_call_and_return_conditional_losses_1121195ù
leaky_re_lu_645/PartitionedCallPartitionedCall8batch_normalization_645/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_645_layer_call_and_return_conditional_losses_1121793
!dense_717/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_645/PartitionedCall:output:0dense_717_1121821dense_717_1121823*
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
GPU 2J 8 *O
fJRH
F__inference_dense_717_layer_call_and_return_conditional_losses_1121820
/batch_normalization_646/StatefulPartitionedCallStatefulPartitionedCall*dense_717/StatefulPartitionedCall:output:0batch_normalization_646_1121826batch_normalization_646_1121828batch_normalization_646_1121830batch_normalization_646_1121832*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_646_layer_call_and_return_conditional_losses_1121277ù
leaky_re_lu_646/PartitionedCallPartitionedCall8batch_normalization_646/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_646_layer_call_and_return_conditional_losses_1121840
!dense_718/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_646/PartitionedCall:output:0dense_718_1121868dense_718_1121870*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_718_layer_call_and_return_conditional_losses_1121867
/batch_normalization_647/StatefulPartitionedCallStatefulPartitionedCall*dense_718/StatefulPartitionedCall:output:0batch_normalization_647_1121873batch_normalization_647_1121875batch_normalization_647_1121877batch_normalization_647_1121879*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_647_layer_call_and_return_conditional_losses_1121359ù
leaky_re_lu_647/PartitionedCallPartitionedCall8batch_normalization_647/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_647_layer_call_and_return_conditional_losses_1121887
!dense_719/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_647/PartitionedCall:output:0dense_719_1121915dense_719_1121917*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_719_layer_call_and_return_conditional_losses_1121914
/batch_normalization_648/StatefulPartitionedCallStatefulPartitionedCall*dense_719/StatefulPartitionedCall:output:0batch_normalization_648_1121920batch_normalization_648_1121922batch_normalization_648_1121924batch_normalization_648_1121926*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_648_layer_call_and_return_conditional_losses_1121441ù
leaky_re_lu_648/PartitionedCallPartitionedCall8batch_normalization_648/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_648_layer_call_and_return_conditional_losses_1121934
!dense_720/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_648/PartitionedCall:output:0dense_720_1121947dense_720_1121949*
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
F__inference_dense_720_layer_call_and_return_conditional_losses_1121946g
"dense_711/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_711/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_711_1121539*
_output_shapes

:i*
dtype0
 dense_711/kernel/Regularizer/AbsAbs7dense_711/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_711/kernel/Regularizer/SumSum$dense_711/kernel/Regularizer/Abs:y:0-dense_711/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_711/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_711/kernel/Regularizer/mulMul+dense_711/kernel/Regularizer/mul/x:output:0)dense_711/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_711/kernel/Regularizer/addAddV2+dense_711/kernel/Regularizer/Const:output:0$dense_711/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_711/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_711_1121539*
_output_shapes

:i*
dtype0
#dense_711/kernel/Regularizer/SquareSquare:dense_711/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_711/kernel/Regularizer/Sum_1Sum'dense_711/kernel/Regularizer/Square:y:0-dense_711/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_711/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_711/kernel/Regularizer/mul_1Mul-dense_711/kernel/Regularizer/mul_1/x:output:0+dense_711/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_711/kernel/Regularizer/add_1AddV2$dense_711/kernel/Regularizer/add:z:0&dense_711/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_712/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_712/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_712_1121586*
_output_shapes

:ii*
dtype0
 dense_712/kernel/Regularizer/AbsAbs7dense_712/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_712/kernel/Regularizer/SumSum$dense_712/kernel/Regularizer/Abs:y:0-dense_712/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_712/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_712/kernel/Regularizer/mulMul+dense_712/kernel/Regularizer/mul/x:output:0)dense_712/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_712/kernel/Regularizer/addAddV2+dense_712/kernel/Regularizer/Const:output:0$dense_712/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_712/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_712_1121586*
_output_shapes

:ii*
dtype0
#dense_712/kernel/Regularizer/SquareSquare:dense_712/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_712/kernel/Regularizer/Sum_1Sum'dense_712/kernel/Regularizer/Square:y:0-dense_712/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_712/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_712/kernel/Regularizer/mul_1Mul-dense_712/kernel/Regularizer/mul_1/x:output:0+dense_712/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_712/kernel/Regularizer/add_1AddV2$dense_712/kernel/Regularizer/add:z:0&dense_712/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_713/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_713/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_713_1121633*
_output_shapes

:ii*
dtype0
 dense_713/kernel/Regularizer/AbsAbs7dense_713/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_713/kernel/Regularizer/SumSum$dense_713/kernel/Regularizer/Abs:y:0-dense_713/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_713/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_713/kernel/Regularizer/mulMul+dense_713/kernel/Regularizer/mul/x:output:0)dense_713/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_713/kernel/Regularizer/addAddV2+dense_713/kernel/Regularizer/Const:output:0$dense_713/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_713/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_713_1121633*
_output_shapes

:ii*
dtype0
#dense_713/kernel/Regularizer/SquareSquare:dense_713/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_713/kernel/Regularizer/Sum_1Sum'dense_713/kernel/Regularizer/Square:y:0-dense_713/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_713/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_713/kernel/Regularizer/mul_1Mul-dense_713/kernel/Regularizer/mul_1/x:output:0+dense_713/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_713/kernel/Regularizer/add_1AddV2$dense_713/kernel/Regularizer/add:z:0&dense_713/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_714/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_714/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_714_1121680*
_output_shapes

:ii*
dtype0
 dense_714/kernel/Regularizer/AbsAbs7dense_714/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_714/kernel/Regularizer/SumSum$dense_714/kernel/Regularizer/Abs:y:0-dense_714/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_714/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_714/kernel/Regularizer/mulMul+dense_714/kernel/Regularizer/mul/x:output:0)dense_714/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_714/kernel/Regularizer/addAddV2+dense_714/kernel/Regularizer/Const:output:0$dense_714/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_714/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_714_1121680*
_output_shapes

:ii*
dtype0
#dense_714/kernel/Regularizer/SquareSquare:dense_714/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_714/kernel/Regularizer/Sum_1Sum'dense_714/kernel/Regularizer/Square:y:0-dense_714/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_714/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_714/kernel/Regularizer/mul_1Mul-dense_714/kernel/Regularizer/mul_1/x:output:0+dense_714/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_714/kernel/Regularizer/add_1AddV2$dense_714/kernel/Regularizer/add:z:0&dense_714/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_715/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_715/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_715_1121727*
_output_shapes

:ii*
dtype0
 dense_715/kernel/Regularizer/AbsAbs7dense_715/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_715/kernel/Regularizer/SumSum$dense_715/kernel/Regularizer/Abs:y:0-dense_715/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_715/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_715/kernel/Regularizer/mulMul+dense_715/kernel/Regularizer/mul/x:output:0)dense_715/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_715/kernel/Regularizer/addAddV2+dense_715/kernel/Regularizer/Const:output:0$dense_715/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_715/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_715_1121727*
_output_shapes

:ii*
dtype0
#dense_715/kernel/Regularizer/SquareSquare:dense_715/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_715/kernel/Regularizer/Sum_1Sum'dense_715/kernel/Regularizer/Square:y:0-dense_715/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_715/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_715/kernel/Regularizer/mul_1Mul-dense_715/kernel/Regularizer/mul_1/x:output:0+dense_715/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_715/kernel/Regularizer/add_1AddV2$dense_715/kernel/Regularizer/add:z:0&dense_715/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_716/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_716/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_716_1121774*
_output_shapes

:i=*
dtype0
 dense_716/kernel/Regularizer/AbsAbs7dense_716/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_716/kernel/Regularizer/SumSum$dense_716/kernel/Regularizer/Abs:y:0-dense_716/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_716/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_716/kernel/Regularizer/mulMul+dense_716/kernel/Regularizer/mul/x:output:0)dense_716/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_716/kernel/Regularizer/addAddV2+dense_716/kernel/Regularizer/Const:output:0$dense_716/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_716/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_716_1121774*
_output_shapes

:i=*
dtype0
#dense_716/kernel/Regularizer/SquareSquare:dense_716/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_716/kernel/Regularizer/Sum_1Sum'dense_716/kernel/Regularizer/Square:y:0-dense_716/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_716/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_716/kernel/Regularizer/mul_1Mul-dense_716/kernel/Regularizer/mul_1/x:output:0+dense_716/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_716/kernel/Regularizer/add_1AddV2$dense_716/kernel/Regularizer/add:z:0&dense_716/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_717/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_717/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_717_1121821*
_output_shapes

:==*
dtype0
 dense_717/kernel/Regularizer/AbsAbs7dense_717/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_717/kernel/Regularizer/SumSum$dense_717/kernel/Regularizer/Abs:y:0-dense_717/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_717/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_717/kernel/Regularizer/mulMul+dense_717/kernel/Regularizer/mul/x:output:0)dense_717/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_717/kernel/Regularizer/addAddV2+dense_717/kernel/Regularizer/Const:output:0$dense_717/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_717/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_717_1121821*
_output_shapes

:==*
dtype0
#dense_717/kernel/Regularizer/SquareSquare:dense_717/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_717/kernel/Regularizer/Sum_1Sum'dense_717/kernel/Regularizer/Square:y:0-dense_717/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_717/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_717/kernel/Regularizer/mul_1Mul-dense_717/kernel/Regularizer/mul_1/x:output:0+dense_717/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_717/kernel/Regularizer/add_1AddV2$dense_717/kernel/Regularizer/add:z:0&dense_717/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_718/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_718/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_718_1121868*
_output_shapes

:=7*
dtype0
 dense_718/kernel/Regularizer/AbsAbs7dense_718/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_718/kernel/Regularizer/SumSum$dense_718/kernel/Regularizer/Abs:y:0-dense_718/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_718/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_718/kernel/Regularizer/mulMul+dense_718/kernel/Regularizer/mul/x:output:0)dense_718/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_718/kernel/Regularizer/addAddV2+dense_718/kernel/Regularizer/Const:output:0$dense_718/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_718/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_718_1121868*
_output_shapes

:=7*
dtype0
#dense_718/kernel/Regularizer/SquareSquare:dense_718/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_718/kernel/Regularizer/Sum_1Sum'dense_718/kernel/Regularizer/Square:y:0-dense_718/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_718/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_718/kernel/Regularizer/mul_1Mul-dense_718/kernel/Regularizer/mul_1/x:output:0+dense_718/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_718/kernel/Regularizer/add_1AddV2$dense_718/kernel/Regularizer/add:z:0&dense_718/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_719/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_719/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_719_1121915*
_output_shapes

:77*
dtype0
 dense_719/kernel/Regularizer/AbsAbs7dense_719/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_719/kernel/Regularizer/SumSum$dense_719/kernel/Regularizer/Abs:y:0-dense_719/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_719/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_719/kernel/Regularizer/mulMul+dense_719/kernel/Regularizer/mul/x:output:0)dense_719/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_719/kernel/Regularizer/addAddV2+dense_719/kernel/Regularizer/Const:output:0$dense_719/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_719/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_719_1121915*
_output_shapes

:77*
dtype0
#dense_719/kernel/Regularizer/SquareSquare:dense_719/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_719/kernel/Regularizer/Sum_1Sum'dense_719/kernel/Regularizer/Square:y:0-dense_719/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_719/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_719/kernel/Regularizer/mul_1Mul-dense_719/kernel/Regularizer/mul_1/x:output:0+dense_719/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_719/kernel/Regularizer/add_1AddV2$dense_719/kernel/Regularizer/add:z:0&dense_719/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_720/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_640/StatefulPartitionedCall0^batch_normalization_641/StatefulPartitionedCall0^batch_normalization_642/StatefulPartitionedCall0^batch_normalization_643/StatefulPartitionedCall0^batch_normalization_644/StatefulPartitionedCall0^batch_normalization_645/StatefulPartitionedCall0^batch_normalization_646/StatefulPartitionedCall0^batch_normalization_647/StatefulPartitionedCall0^batch_normalization_648/StatefulPartitionedCall"^dense_711/StatefulPartitionedCall0^dense_711/kernel/Regularizer/Abs/ReadVariableOp3^dense_711/kernel/Regularizer/Square/ReadVariableOp"^dense_712/StatefulPartitionedCall0^dense_712/kernel/Regularizer/Abs/ReadVariableOp3^dense_712/kernel/Regularizer/Square/ReadVariableOp"^dense_713/StatefulPartitionedCall0^dense_713/kernel/Regularizer/Abs/ReadVariableOp3^dense_713/kernel/Regularizer/Square/ReadVariableOp"^dense_714/StatefulPartitionedCall0^dense_714/kernel/Regularizer/Abs/ReadVariableOp3^dense_714/kernel/Regularizer/Square/ReadVariableOp"^dense_715/StatefulPartitionedCall0^dense_715/kernel/Regularizer/Abs/ReadVariableOp3^dense_715/kernel/Regularizer/Square/ReadVariableOp"^dense_716/StatefulPartitionedCall0^dense_716/kernel/Regularizer/Abs/ReadVariableOp3^dense_716/kernel/Regularizer/Square/ReadVariableOp"^dense_717/StatefulPartitionedCall0^dense_717/kernel/Regularizer/Abs/ReadVariableOp3^dense_717/kernel/Regularizer/Square/ReadVariableOp"^dense_718/StatefulPartitionedCall0^dense_718/kernel/Regularizer/Abs/ReadVariableOp3^dense_718/kernel/Regularizer/Square/ReadVariableOp"^dense_719/StatefulPartitionedCall0^dense_719/kernel/Regularizer/Abs/ReadVariableOp3^dense_719/kernel/Regularizer/Square/ReadVariableOp"^dense_720/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_640/StatefulPartitionedCall/batch_normalization_640/StatefulPartitionedCall2b
/batch_normalization_641/StatefulPartitionedCall/batch_normalization_641/StatefulPartitionedCall2b
/batch_normalization_642/StatefulPartitionedCall/batch_normalization_642/StatefulPartitionedCall2b
/batch_normalization_643/StatefulPartitionedCall/batch_normalization_643/StatefulPartitionedCall2b
/batch_normalization_644/StatefulPartitionedCall/batch_normalization_644/StatefulPartitionedCall2b
/batch_normalization_645/StatefulPartitionedCall/batch_normalization_645/StatefulPartitionedCall2b
/batch_normalization_646/StatefulPartitionedCall/batch_normalization_646/StatefulPartitionedCall2b
/batch_normalization_647/StatefulPartitionedCall/batch_normalization_647/StatefulPartitionedCall2b
/batch_normalization_648/StatefulPartitionedCall/batch_normalization_648/StatefulPartitionedCall2F
!dense_711/StatefulPartitionedCall!dense_711/StatefulPartitionedCall2b
/dense_711/kernel/Regularizer/Abs/ReadVariableOp/dense_711/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_711/kernel/Regularizer/Square/ReadVariableOp2dense_711/kernel/Regularizer/Square/ReadVariableOp2F
!dense_712/StatefulPartitionedCall!dense_712/StatefulPartitionedCall2b
/dense_712/kernel/Regularizer/Abs/ReadVariableOp/dense_712/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_712/kernel/Regularizer/Square/ReadVariableOp2dense_712/kernel/Regularizer/Square/ReadVariableOp2F
!dense_713/StatefulPartitionedCall!dense_713/StatefulPartitionedCall2b
/dense_713/kernel/Regularizer/Abs/ReadVariableOp/dense_713/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_713/kernel/Regularizer/Square/ReadVariableOp2dense_713/kernel/Regularizer/Square/ReadVariableOp2F
!dense_714/StatefulPartitionedCall!dense_714/StatefulPartitionedCall2b
/dense_714/kernel/Regularizer/Abs/ReadVariableOp/dense_714/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_714/kernel/Regularizer/Square/ReadVariableOp2dense_714/kernel/Regularizer/Square/ReadVariableOp2F
!dense_715/StatefulPartitionedCall!dense_715/StatefulPartitionedCall2b
/dense_715/kernel/Regularizer/Abs/ReadVariableOp/dense_715/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_715/kernel/Regularizer/Square/ReadVariableOp2dense_715/kernel/Regularizer/Square/ReadVariableOp2F
!dense_716/StatefulPartitionedCall!dense_716/StatefulPartitionedCall2b
/dense_716/kernel/Regularizer/Abs/ReadVariableOp/dense_716/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_716/kernel/Regularizer/Square/ReadVariableOp2dense_716/kernel/Regularizer/Square/ReadVariableOp2F
!dense_717/StatefulPartitionedCall!dense_717/StatefulPartitionedCall2b
/dense_717/kernel/Regularizer/Abs/ReadVariableOp/dense_717/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_717/kernel/Regularizer/Square/ReadVariableOp2dense_717/kernel/Regularizer/Square/ReadVariableOp2F
!dense_718/StatefulPartitionedCall!dense_718/StatefulPartitionedCall2b
/dense_718/kernel/Regularizer/Abs/ReadVariableOp/dense_718/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_718/kernel/Regularizer/Square/ReadVariableOp2dense_718/kernel/Regularizer/Square/ReadVariableOp2F
!dense_719/StatefulPartitionedCall!dense_719/StatefulPartitionedCall2b
/dense_719/kernel/Regularizer/Abs/ReadVariableOp/dense_719/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_719/kernel/Regularizer/Square/ReadVariableOp2dense_719/kernel/Regularizer/Square/ReadVariableOp2F
!dense_720/StatefulPartitionedCall!dense_720/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ñ
³
T__inference_batch_normalization_644_layer_call_and_return_conditional_losses_1125628

inputs/
!batchnorm_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i1
#batchnorm_readvariableop_1_resource:i1
#batchnorm_readvariableop_2_resource:i
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:iz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs

ã
__inference_loss_fn_2_1126307J
8dense_713_kernel_regularizer_abs_readvariableop_resource:ii
identity¢/dense_713/kernel/Regularizer/Abs/ReadVariableOp¢2dense_713/kernel/Regularizer/Square/ReadVariableOpg
"dense_713/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
/dense_713/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8dense_713_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_713/kernel/Regularizer/AbsAbs7dense_713/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_713/kernel/Regularizer/SumSum$dense_713/kernel/Regularizer/Abs:y:0-dense_713/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_713/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_713/kernel/Regularizer/mulMul+dense_713/kernel/Regularizer/mul/x:output:0)dense_713/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_713/kernel/Regularizer/addAddV2+dense_713/kernel/Regularizer/Const:output:0$dense_713/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: «
2dense_713/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense_713_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_713/kernel/Regularizer/SquareSquare:dense_713/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_713/kernel/Regularizer/Sum_1Sum'dense_713/kernel/Regularizer/Square:y:0-dense_713/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_713/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_713/kernel/Regularizer/mul_1Mul-dense_713/kernel/Regularizer/mul_1/x:output:0+dense_713/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_713/kernel/Regularizer/add_1AddV2$dense_713/kernel/Regularizer/add:z:0&dense_713/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
IdentityIdentity&dense_713/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ­
NoOpNoOp0^dense_713/kernel/Regularizer/Abs/ReadVariableOp3^dense_713/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_713/kernel/Regularizer/Abs/ReadVariableOp/dense_713/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_713/kernel/Regularizer/Square/ReadVariableOp2dense_713/kernel/Regularizer/Square/ReadVariableOp

ã
__inference_loss_fn_4_1126347J
8dense_715_kernel_regularizer_abs_readvariableop_resource:ii
identity¢/dense_715/kernel/Regularizer/Abs/ReadVariableOp¢2dense_715/kernel/Regularizer/Square/ReadVariableOpg
"dense_715/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
/dense_715/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8dense_715_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_715/kernel/Regularizer/AbsAbs7dense_715/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_715/kernel/Regularizer/SumSum$dense_715/kernel/Regularizer/Abs:y:0-dense_715/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_715/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_715/kernel/Regularizer/mulMul+dense_715/kernel/Regularizer/mul/x:output:0)dense_715/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_715/kernel/Regularizer/addAddV2+dense_715/kernel/Regularizer/Const:output:0$dense_715/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: «
2dense_715/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense_715_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_715/kernel/Regularizer/SquareSquare:dense_715/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_715/kernel/Regularizer/Sum_1Sum'dense_715/kernel/Regularizer/Square:y:0-dense_715/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_715/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_715/kernel/Regularizer/mul_1Mul-dense_715/kernel/Regularizer/mul_1/x:output:0+dense_715/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_715/kernel/Regularizer/add_1AddV2$dense_715/kernel/Regularizer/add:z:0&dense_715/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
IdentityIdentity&dense_715/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ­
NoOpNoOp0^dense_715/kernel/Regularizer/Abs/ReadVariableOp3^dense_715/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_715/kernel/Regularizer/Abs/ReadVariableOp/dense_715/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_715/kernel/Regularizer/Square/ReadVariableOp2dense_715/kernel/Regularizer/Square/ReadVariableOp
æ
h
L__inference_leaky_re_lu_647_layer_call_and_return_conditional_losses_1121887

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_642_layer_call_and_return_conditional_losses_1125394

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿi:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_644_layer_call_and_return_conditional_losses_1121746

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿi:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_645_layer_call_and_return_conditional_losses_1125801

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
¥
Þ
F__inference_dense_718_layer_call_and_return_conditional_losses_1121867

inputs0
matmul_readvariableop_resource:=7-
biasadd_readvariableop_resource:7
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_718/kernel/Regularizer/Abs/ReadVariableOp¢2dense_718/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:=7*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:7*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7g
"dense_718/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_718/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:=7*
dtype0
 dense_718/kernel/Regularizer/AbsAbs7dense_718/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_718/kernel/Regularizer/SumSum$dense_718/kernel/Regularizer/Abs:y:0-dense_718/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_718/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_718/kernel/Regularizer/mulMul+dense_718/kernel/Regularizer/mul/x:output:0)dense_718/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_718/kernel/Regularizer/addAddV2+dense_718/kernel/Regularizer/Const:output:0$dense_718/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_718/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:=7*
dtype0
#dense_718/kernel/Regularizer/SquareSquare:dense_718/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_718/kernel/Regularizer/Sum_1Sum'dense_718/kernel/Regularizer/Square:y:0-dense_718/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_718/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_718/kernel/Regularizer/mul_1Mul-dense_718/kernel/Regularizer/mul_1/x:output:0+dense_718/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_718/kernel/Regularizer/add_1AddV2$dense_718/kernel/Regularizer/add:z:0&dense_718/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7Þ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_718/kernel/Regularizer/Abs/ReadVariableOp3^dense_718/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_718/kernel/Regularizer/Abs/ReadVariableOp/dense_718/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_718/kernel/Regularizer/Square/ReadVariableOp2dense_718/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_645_layer_call_and_return_conditional_losses_1121195

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
æ
h
L__inference_leaky_re_lu_647_layer_call_and_return_conditional_losses_1126089

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_642_layer_call_fn_1125389

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
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_642_layer_call_and_return_conditional_losses_1121652`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿi:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_640_layer_call_and_return_conditional_losses_1120785

inputs/
!batchnorm_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i1
#batchnorm_readvariableop_1_resource:i1
#batchnorm_readvariableop_2_resource:i
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:iz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
õ
;
J__inference_sequential_71_layer_call_and_return_conditional_losses_1124322

inputs
normalization_71_sub_y
normalization_71_sqrt_x:
(dense_711_matmul_readvariableop_resource:i7
)dense_711_biasadd_readvariableop_resource:iG
9batch_normalization_640_batchnorm_readvariableop_resource:iK
=batch_normalization_640_batchnorm_mul_readvariableop_resource:iI
;batch_normalization_640_batchnorm_readvariableop_1_resource:iI
;batch_normalization_640_batchnorm_readvariableop_2_resource:i:
(dense_712_matmul_readvariableop_resource:ii7
)dense_712_biasadd_readvariableop_resource:iG
9batch_normalization_641_batchnorm_readvariableop_resource:iK
=batch_normalization_641_batchnorm_mul_readvariableop_resource:iI
;batch_normalization_641_batchnorm_readvariableop_1_resource:iI
;batch_normalization_641_batchnorm_readvariableop_2_resource:i:
(dense_713_matmul_readvariableop_resource:ii7
)dense_713_biasadd_readvariableop_resource:iG
9batch_normalization_642_batchnorm_readvariableop_resource:iK
=batch_normalization_642_batchnorm_mul_readvariableop_resource:iI
;batch_normalization_642_batchnorm_readvariableop_1_resource:iI
;batch_normalization_642_batchnorm_readvariableop_2_resource:i:
(dense_714_matmul_readvariableop_resource:ii7
)dense_714_biasadd_readvariableop_resource:iG
9batch_normalization_643_batchnorm_readvariableop_resource:iK
=batch_normalization_643_batchnorm_mul_readvariableop_resource:iI
;batch_normalization_643_batchnorm_readvariableop_1_resource:iI
;batch_normalization_643_batchnorm_readvariableop_2_resource:i:
(dense_715_matmul_readvariableop_resource:ii7
)dense_715_biasadd_readvariableop_resource:iG
9batch_normalization_644_batchnorm_readvariableop_resource:iK
=batch_normalization_644_batchnorm_mul_readvariableop_resource:iI
;batch_normalization_644_batchnorm_readvariableop_1_resource:iI
;batch_normalization_644_batchnorm_readvariableop_2_resource:i:
(dense_716_matmul_readvariableop_resource:i=7
)dense_716_biasadd_readvariableop_resource:=G
9batch_normalization_645_batchnorm_readvariableop_resource:=K
=batch_normalization_645_batchnorm_mul_readvariableop_resource:=I
;batch_normalization_645_batchnorm_readvariableop_1_resource:=I
;batch_normalization_645_batchnorm_readvariableop_2_resource:=:
(dense_717_matmul_readvariableop_resource:==7
)dense_717_biasadd_readvariableop_resource:=G
9batch_normalization_646_batchnorm_readvariableop_resource:=K
=batch_normalization_646_batchnorm_mul_readvariableop_resource:=I
;batch_normalization_646_batchnorm_readvariableop_1_resource:=I
;batch_normalization_646_batchnorm_readvariableop_2_resource:=:
(dense_718_matmul_readvariableop_resource:=77
)dense_718_biasadd_readvariableop_resource:7G
9batch_normalization_647_batchnorm_readvariableop_resource:7K
=batch_normalization_647_batchnorm_mul_readvariableop_resource:7I
;batch_normalization_647_batchnorm_readvariableop_1_resource:7I
;batch_normalization_647_batchnorm_readvariableop_2_resource:7:
(dense_719_matmul_readvariableop_resource:777
)dense_719_biasadd_readvariableop_resource:7G
9batch_normalization_648_batchnorm_readvariableop_resource:7K
=batch_normalization_648_batchnorm_mul_readvariableop_resource:7I
;batch_normalization_648_batchnorm_readvariableop_1_resource:7I
;batch_normalization_648_batchnorm_readvariableop_2_resource:7:
(dense_720_matmul_readvariableop_resource:77
)dense_720_biasadd_readvariableop_resource:
identity¢0batch_normalization_640/batchnorm/ReadVariableOp¢2batch_normalization_640/batchnorm/ReadVariableOp_1¢2batch_normalization_640/batchnorm/ReadVariableOp_2¢4batch_normalization_640/batchnorm/mul/ReadVariableOp¢0batch_normalization_641/batchnorm/ReadVariableOp¢2batch_normalization_641/batchnorm/ReadVariableOp_1¢2batch_normalization_641/batchnorm/ReadVariableOp_2¢4batch_normalization_641/batchnorm/mul/ReadVariableOp¢0batch_normalization_642/batchnorm/ReadVariableOp¢2batch_normalization_642/batchnorm/ReadVariableOp_1¢2batch_normalization_642/batchnorm/ReadVariableOp_2¢4batch_normalization_642/batchnorm/mul/ReadVariableOp¢0batch_normalization_643/batchnorm/ReadVariableOp¢2batch_normalization_643/batchnorm/ReadVariableOp_1¢2batch_normalization_643/batchnorm/ReadVariableOp_2¢4batch_normalization_643/batchnorm/mul/ReadVariableOp¢0batch_normalization_644/batchnorm/ReadVariableOp¢2batch_normalization_644/batchnorm/ReadVariableOp_1¢2batch_normalization_644/batchnorm/ReadVariableOp_2¢4batch_normalization_644/batchnorm/mul/ReadVariableOp¢0batch_normalization_645/batchnorm/ReadVariableOp¢2batch_normalization_645/batchnorm/ReadVariableOp_1¢2batch_normalization_645/batchnorm/ReadVariableOp_2¢4batch_normalization_645/batchnorm/mul/ReadVariableOp¢0batch_normalization_646/batchnorm/ReadVariableOp¢2batch_normalization_646/batchnorm/ReadVariableOp_1¢2batch_normalization_646/batchnorm/ReadVariableOp_2¢4batch_normalization_646/batchnorm/mul/ReadVariableOp¢0batch_normalization_647/batchnorm/ReadVariableOp¢2batch_normalization_647/batchnorm/ReadVariableOp_1¢2batch_normalization_647/batchnorm/ReadVariableOp_2¢4batch_normalization_647/batchnorm/mul/ReadVariableOp¢0batch_normalization_648/batchnorm/ReadVariableOp¢2batch_normalization_648/batchnorm/ReadVariableOp_1¢2batch_normalization_648/batchnorm/ReadVariableOp_2¢4batch_normalization_648/batchnorm/mul/ReadVariableOp¢ dense_711/BiasAdd/ReadVariableOp¢dense_711/MatMul/ReadVariableOp¢/dense_711/kernel/Regularizer/Abs/ReadVariableOp¢2dense_711/kernel/Regularizer/Square/ReadVariableOp¢ dense_712/BiasAdd/ReadVariableOp¢dense_712/MatMul/ReadVariableOp¢/dense_712/kernel/Regularizer/Abs/ReadVariableOp¢2dense_712/kernel/Regularizer/Square/ReadVariableOp¢ dense_713/BiasAdd/ReadVariableOp¢dense_713/MatMul/ReadVariableOp¢/dense_713/kernel/Regularizer/Abs/ReadVariableOp¢2dense_713/kernel/Regularizer/Square/ReadVariableOp¢ dense_714/BiasAdd/ReadVariableOp¢dense_714/MatMul/ReadVariableOp¢/dense_714/kernel/Regularizer/Abs/ReadVariableOp¢2dense_714/kernel/Regularizer/Square/ReadVariableOp¢ dense_715/BiasAdd/ReadVariableOp¢dense_715/MatMul/ReadVariableOp¢/dense_715/kernel/Regularizer/Abs/ReadVariableOp¢2dense_715/kernel/Regularizer/Square/ReadVariableOp¢ dense_716/BiasAdd/ReadVariableOp¢dense_716/MatMul/ReadVariableOp¢/dense_716/kernel/Regularizer/Abs/ReadVariableOp¢2dense_716/kernel/Regularizer/Square/ReadVariableOp¢ dense_717/BiasAdd/ReadVariableOp¢dense_717/MatMul/ReadVariableOp¢/dense_717/kernel/Regularizer/Abs/ReadVariableOp¢2dense_717/kernel/Regularizer/Square/ReadVariableOp¢ dense_718/BiasAdd/ReadVariableOp¢dense_718/MatMul/ReadVariableOp¢/dense_718/kernel/Regularizer/Abs/ReadVariableOp¢2dense_718/kernel/Regularizer/Square/ReadVariableOp¢ dense_719/BiasAdd/ReadVariableOp¢dense_719/MatMul/ReadVariableOp¢/dense_719/kernel/Regularizer/Abs/ReadVariableOp¢2dense_719/kernel/Regularizer/Square/ReadVariableOp¢ dense_720/BiasAdd/ReadVariableOp¢dense_720/MatMul/ReadVariableOpm
normalization_71/subSubinputsnormalization_71_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_71/SqrtSqrtnormalization_71_sqrt_x*
T0*
_output_shapes

:_
normalization_71/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_71/MaximumMaximumnormalization_71/Sqrt:y:0#normalization_71/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_71/truedivRealDivnormalization_71/sub:z:0normalization_71/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_711/MatMul/ReadVariableOpReadVariableOp(dense_711_matmul_readvariableop_resource*
_output_shapes

:i*
dtype0
dense_711/MatMulMatMulnormalization_71/truediv:z:0'dense_711/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 dense_711/BiasAdd/ReadVariableOpReadVariableOp)dense_711_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
dense_711/BiasAddBiasAdddense_711/MatMul:product:0(dense_711/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi¦
0batch_normalization_640/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_640_batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0l
'batch_normalization_640/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_640/batchnorm/addAddV28batch_normalization_640/batchnorm/ReadVariableOp:value:00batch_normalization_640/batchnorm/add/y:output:0*
T0*
_output_shapes
:i
'batch_normalization_640/batchnorm/RsqrtRsqrt)batch_normalization_640/batchnorm/add:z:0*
T0*
_output_shapes
:i®
4batch_normalization_640/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_640_batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0¼
%batch_normalization_640/batchnorm/mulMul+batch_normalization_640/batchnorm/Rsqrt:y:0<batch_normalization_640/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:i§
'batch_normalization_640/batchnorm/mul_1Muldense_711/BiasAdd:output:0)batch_normalization_640/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiª
2batch_normalization_640/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_640_batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0º
'batch_normalization_640/batchnorm/mul_2Mul:batch_normalization_640/batchnorm/ReadVariableOp_1:value:0)batch_normalization_640/batchnorm/mul:z:0*
T0*
_output_shapes
:iª
2batch_normalization_640/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_640_batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0º
%batch_normalization_640/batchnorm/subSub:batch_normalization_640/batchnorm/ReadVariableOp_2:value:0+batch_normalization_640/batchnorm/mul_2:z:0*
T0*
_output_shapes
:iº
'batch_normalization_640/batchnorm/add_1AddV2+batch_normalization_640/batchnorm/mul_1:z:0)batch_normalization_640/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
leaky_re_lu_640/LeakyRelu	LeakyRelu+batch_normalization_640/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>
dense_712/MatMul/ReadVariableOpReadVariableOp(dense_712_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
dense_712/MatMulMatMul'leaky_re_lu_640/LeakyRelu:activations:0'dense_712/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 dense_712/BiasAdd/ReadVariableOpReadVariableOp)dense_712_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
dense_712/BiasAddBiasAdddense_712/MatMul:product:0(dense_712/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi¦
0batch_normalization_641/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_641_batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0l
'batch_normalization_641/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_641/batchnorm/addAddV28batch_normalization_641/batchnorm/ReadVariableOp:value:00batch_normalization_641/batchnorm/add/y:output:0*
T0*
_output_shapes
:i
'batch_normalization_641/batchnorm/RsqrtRsqrt)batch_normalization_641/batchnorm/add:z:0*
T0*
_output_shapes
:i®
4batch_normalization_641/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_641_batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0¼
%batch_normalization_641/batchnorm/mulMul+batch_normalization_641/batchnorm/Rsqrt:y:0<batch_normalization_641/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:i§
'batch_normalization_641/batchnorm/mul_1Muldense_712/BiasAdd:output:0)batch_normalization_641/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiª
2batch_normalization_641/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_641_batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0º
'batch_normalization_641/batchnorm/mul_2Mul:batch_normalization_641/batchnorm/ReadVariableOp_1:value:0)batch_normalization_641/batchnorm/mul:z:0*
T0*
_output_shapes
:iª
2batch_normalization_641/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_641_batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0º
%batch_normalization_641/batchnorm/subSub:batch_normalization_641/batchnorm/ReadVariableOp_2:value:0+batch_normalization_641/batchnorm/mul_2:z:0*
T0*
_output_shapes
:iº
'batch_normalization_641/batchnorm/add_1AddV2+batch_normalization_641/batchnorm/mul_1:z:0)batch_normalization_641/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
leaky_re_lu_641/LeakyRelu	LeakyRelu+batch_normalization_641/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>
dense_713/MatMul/ReadVariableOpReadVariableOp(dense_713_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
dense_713/MatMulMatMul'leaky_re_lu_641/LeakyRelu:activations:0'dense_713/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 dense_713/BiasAdd/ReadVariableOpReadVariableOp)dense_713_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
dense_713/BiasAddBiasAdddense_713/MatMul:product:0(dense_713/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi¦
0batch_normalization_642/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_642_batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0l
'batch_normalization_642/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_642/batchnorm/addAddV28batch_normalization_642/batchnorm/ReadVariableOp:value:00batch_normalization_642/batchnorm/add/y:output:0*
T0*
_output_shapes
:i
'batch_normalization_642/batchnorm/RsqrtRsqrt)batch_normalization_642/batchnorm/add:z:0*
T0*
_output_shapes
:i®
4batch_normalization_642/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_642_batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0¼
%batch_normalization_642/batchnorm/mulMul+batch_normalization_642/batchnorm/Rsqrt:y:0<batch_normalization_642/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:i§
'batch_normalization_642/batchnorm/mul_1Muldense_713/BiasAdd:output:0)batch_normalization_642/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiª
2batch_normalization_642/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_642_batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0º
'batch_normalization_642/batchnorm/mul_2Mul:batch_normalization_642/batchnorm/ReadVariableOp_1:value:0)batch_normalization_642/batchnorm/mul:z:0*
T0*
_output_shapes
:iª
2batch_normalization_642/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_642_batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0º
%batch_normalization_642/batchnorm/subSub:batch_normalization_642/batchnorm/ReadVariableOp_2:value:0+batch_normalization_642/batchnorm/mul_2:z:0*
T0*
_output_shapes
:iº
'batch_normalization_642/batchnorm/add_1AddV2+batch_normalization_642/batchnorm/mul_1:z:0)batch_normalization_642/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
leaky_re_lu_642/LeakyRelu	LeakyRelu+batch_normalization_642/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>
dense_714/MatMul/ReadVariableOpReadVariableOp(dense_714_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
dense_714/MatMulMatMul'leaky_re_lu_642/LeakyRelu:activations:0'dense_714/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 dense_714/BiasAdd/ReadVariableOpReadVariableOp)dense_714_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
dense_714/BiasAddBiasAdddense_714/MatMul:product:0(dense_714/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi¦
0batch_normalization_643/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_643_batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0l
'batch_normalization_643/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_643/batchnorm/addAddV28batch_normalization_643/batchnorm/ReadVariableOp:value:00batch_normalization_643/batchnorm/add/y:output:0*
T0*
_output_shapes
:i
'batch_normalization_643/batchnorm/RsqrtRsqrt)batch_normalization_643/batchnorm/add:z:0*
T0*
_output_shapes
:i®
4batch_normalization_643/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_643_batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0¼
%batch_normalization_643/batchnorm/mulMul+batch_normalization_643/batchnorm/Rsqrt:y:0<batch_normalization_643/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:i§
'batch_normalization_643/batchnorm/mul_1Muldense_714/BiasAdd:output:0)batch_normalization_643/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiª
2batch_normalization_643/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_643_batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0º
'batch_normalization_643/batchnorm/mul_2Mul:batch_normalization_643/batchnorm/ReadVariableOp_1:value:0)batch_normalization_643/batchnorm/mul:z:0*
T0*
_output_shapes
:iª
2batch_normalization_643/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_643_batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0º
%batch_normalization_643/batchnorm/subSub:batch_normalization_643/batchnorm/ReadVariableOp_2:value:0+batch_normalization_643/batchnorm/mul_2:z:0*
T0*
_output_shapes
:iº
'batch_normalization_643/batchnorm/add_1AddV2+batch_normalization_643/batchnorm/mul_1:z:0)batch_normalization_643/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
leaky_re_lu_643/LeakyRelu	LeakyRelu+batch_normalization_643/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>
dense_715/MatMul/ReadVariableOpReadVariableOp(dense_715_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
dense_715/MatMulMatMul'leaky_re_lu_643/LeakyRelu:activations:0'dense_715/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 dense_715/BiasAdd/ReadVariableOpReadVariableOp)dense_715_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
dense_715/BiasAddBiasAdddense_715/MatMul:product:0(dense_715/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi¦
0batch_normalization_644/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_644_batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0l
'batch_normalization_644/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_644/batchnorm/addAddV28batch_normalization_644/batchnorm/ReadVariableOp:value:00batch_normalization_644/batchnorm/add/y:output:0*
T0*
_output_shapes
:i
'batch_normalization_644/batchnorm/RsqrtRsqrt)batch_normalization_644/batchnorm/add:z:0*
T0*
_output_shapes
:i®
4batch_normalization_644/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_644_batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0¼
%batch_normalization_644/batchnorm/mulMul+batch_normalization_644/batchnorm/Rsqrt:y:0<batch_normalization_644/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:i§
'batch_normalization_644/batchnorm/mul_1Muldense_715/BiasAdd:output:0)batch_normalization_644/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiª
2batch_normalization_644/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_644_batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0º
'batch_normalization_644/batchnorm/mul_2Mul:batch_normalization_644/batchnorm/ReadVariableOp_1:value:0)batch_normalization_644/batchnorm/mul:z:0*
T0*
_output_shapes
:iª
2batch_normalization_644/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_644_batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0º
%batch_normalization_644/batchnorm/subSub:batch_normalization_644/batchnorm/ReadVariableOp_2:value:0+batch_normalization_644/batchnorm/mul_2:z:0*
T0*
_output_shapes
:iº
'batch_normalization_644/batchnorm/add_1AddV2+batch_normalization_644/batchnorm/mul_1:z:0)batch_normalization_644/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
leaky_re_lu_644/LeakyRelu	LeakyRelu+batch_normalization_644/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>
dense_716/MatMul/ReadVariableOpReadVariableOp(dense_716_matmul_readvariableop_resource*
_output_shapes

:i=*
dtype0
dense_716/MatMulMatMul'leaky_re_lu_644/LeakyRelu:activations:0'dense_716/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 dense_716/BiasAdd/ReadVariableOpReadVariableOp)dense_716_biasadd_readvariableop_resource*
_output_shapes
:=*
dtype0
dense_716/BiasAddBiasAdddense_716/MatMul:product:0(dense_716/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=¦
0batch_normalization_645/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_645_batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0l
'batch_normalization_645/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_645/batchnorm/addAddV28batch_normalization_645/batchnorm/ReadVariableOp:value:00batch_normalization_645/batchnorm/add/y:output:0*
T0*
_output_shapes
:=
'batch_normalization_645/batchnorm/RsqrtRsqrt)batch_normalization_645/batchnorm/add:z:0*
T0*
_output_shapes
:=®
4batch_normalization_645/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_645_batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0¼
%batch_normalization_645/batchnorm/mulMul+batch_normalization_645/batchnorm/Rsqrt:y:0<batch_normalization_645/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=§
'batch_normalization_645/batchnorm/mul_1Muldense_716/BiasAdd:output:0)batch_normalization_645/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=ª
2batch_normalization_645/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_645_batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0º
'batch_normalization_645/batchnorm/mul_2Mul:batch_normalization_645/batchnorm/ReadVariableOp_1:value:0)batch_normalization_645/batchnorm/mul:z:0*
T0*
_output_shapes
:=ª
2batch_normalization_645/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_645_batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0º
%batch_normalization_645/batchnorm/subSub:batch_normalization_645/batchnorm/ReadVariableOp_2:value:0+batch_normalization_645/batchnorm/mul_2:z:0*
T0*
_output_shapes
:=º
'batch_normalization_645/batchnorm/add_1AddV2+batch_normalization_645/batchnorm/mul_1:z:0)batch_normalization_645/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
leaky_re_lu_645/LeakyRelu	LeakyRelu+batch_normalization_645/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>
dense_717/MatMul/ReadVariableOpReadVariableOp(dense_717_matmul_readvariableop_resource*
_output_shapes

:==*
dtype0
dense_717/MatMulMatMul'leaky_re_lu_645/LeakyRelu:activations:0'dense_717/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 dense_717/BiasAdd/ReadVariableOpReadVariableOp)dense_717_biasadd_readvariableop_resource*
_output_shapes
:=*
dtype0
dense_717/BiasAddBiasAdddense_717/MatMul:product:0(dense_717/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=¦
0batch_normalization_646/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_646_batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0l
'batch_normalization_646/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_646/batchnorm/addAddV28batch_normalization_646/batchnorm/ReadVariableOp:value:00batch_normalization_646/batchnorm/add/y:output:0*
T0*
_output_shapes
:=
'batch_normalization_646/batchnorm/RsqrtRsqrt)batch_normalization_646/batchnorm/add:z:0*
T0*
_output_shapes
:=®
4batch_normalization_646/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_646_batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0¼
%batch_normalization_646/batchnorm/mulMul+batch_normalization_646/batchnorm/Rsqrt:y:0<batch_normalization_646/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=§
'batch_normalization_646/batchnorm/mul_1Muldense_717/BiasAdd:output:0)batch_normalization_646/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=ª
2batch_normalization_646/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_646_batchnorm_readvariableop_1_resource*
_output_shapes
:=*
dtype0º
'batch_normalization_646/batchnorm/mul_2Mul:batch_normalization_646/batchnorm/ReadVariableOp_1:value:0)batch_normalization_646/batchnorm/mul:z:0*
T0*
_output_shapes
:=ª
2batch_normalization_646/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_646_batchnorm_readvariableop_2_resource*
_output_shapes
:=*
dtype0º
%batch_normalization_646/batchnorm/subSub:batch_normalization_646/batchnorm/ReadVariableOp_2:value:0+batch_normalization_646/batchnorm/mul_2:z:0*
T0*
_output_shapes
:=º
'batch_normalization_646/batchnorm/add_1AddV2+batch_normalization_646/batchnorm/mul_1:z:0)batch_normalization_646/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
leaky_re_lu_646/LeakyRelu	LeakyRelu+batch_normalization_646/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>
dense_718/MatMul/ReadVariableOpReadVariableOp(dense_718_matmul_readvariableop_resource*
_output_shapes

:=7*
dtype0
dense_718/MatMulMatMul'leaky_re_lu_646/LeakyRelu:activations:0'dense_718/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 dense_718/BiasAdd/ReadVariableOpReadVariableOp)dense_718_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0
dense_718/BiasAddBiasAdddense_718/MatMul:product:0(dense_718/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¦
0batch_normalization_647/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_647_batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0l
'batch_normalization_647/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_647/batchnorm/addAddV28batch_normalization_647/batchnorm/ReadVariableOp:value:00batch_normalization_647/batchnorm/add/y:output:0*
T0*
_output_shapes
:7
'batch_normalization_647/batchnorm/RsqrtRsqrt)batch_normalization_647/batchnorm/add:z:0*
T0*
_output_shapes
:7®
4batch_normalization_647/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_647_batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0¼
%batch_normalization_647/batchnorm/mulMul+batch_normalization_647/batchnorm/Rsqrt:y:0<batch_normalization_647/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7§
'batch_normalization_647/batchnorm/mul_1Muldense_718/BiasAdd:output:0)batch_normalization_647/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7ª
2batch_normalization_647/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_647_batchnorm_readvariableop_1_resource*
_output_shapes
:7*
dtype0º
'batch_normalization_647/batchnorm/mul_2Mul:batch_normalization_647/batchnorm/ReadVariableOp_1:value:0)batch_normalization_647/batchnorm/mul:z:0*
T0*
_output_shapes
:7ª
2batch_normalization_647/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_647_batchnorm_readvariableop_2_resource*
_output_shapes
:7*
dtype0º
%batch_normalization_647/batchnorm/subSub:batch_normalization_647/batchnorm/ReadVariableOp_2:value:0+batch_normalization_647/batchnorm/mul_2:z:0*
T0*
_output_shapes
:7º
'batch_normalization_647/batchnorm/add_1AddV2+batch_normalization_647/batchnorm/mul_1:z:0)batch_normalization_647/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
leaky_re_lu_647/LeakyRelu	LeakyRelu+batch_normalization_647/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>
dense_719/MatMul/ReadVariableOpReadVariableOp(dense_719_matmul_readvariableop_resource*
_output_shapes

:77*
dtype0
dense_719/MatMulMatMul'leaky_re_lu_647/LeakyRelu:activations:0'dense_719/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 dense_719/BiasAdd/ReadVariableOpReadVariableOp)dense_719_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0
dense_719/BiasAddBiasAdddense_719/MatMul:product:0(dense_719/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¦
0batch_normalization_648/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_648_batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0l
'batch_normalization_648/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_648/batchnorm/addAddV28batch_normalization_648/batchnorm/ReadVariableOp:value:00batch_normalization_648/batchnorm/add/y:output:0*
T0*
_output_shapes
:7
'batch_normalization_648/batchnorm/RsqrtRsqrt)batch_normalization_648/batchnorm/add:z:0*
T0*
_output_shapes
:7®
4batch_normalization_648/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_648_batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0¼
%batch_normalization_648/batchnorm/mulMul+batch_normalization_648/batchnorm/Rsqrt:y:0<batch_normalization_648/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7§
'batch_normalization_648/batchnorm/mul_1Muldense_719/BiasAdd:output:0)batch_normalization_648/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7ª
2batch_normalization_648/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_648_batchnorm_readvariableop_1_resource*
_output_shapes
:7*
dtype0º
'batch_normalization_648/batchnorm/mul_2Mul:batch_normalization_648/batchnorm/ReadVariableOp_1:value:0)batch_normalization_648/batchnorm/mul:z:0*
T0*
_output_shapes
:7ª
2batch_normalization_648/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_648_batchnorm_readvariableop_2_resource*
_output_shapes
:7*
dtype0º
%batch_normalization_648/batchnorm/subSub:batch_normalization_648/batchnorm/ReadVariableOp_2:value:0+batch_normalization_648/batchnorm/mul_2:z:0*
T0*
_output_shapes
:7º
'batch_normalization_648/batchnorm/add_1AddV2+batch_normalization_648/batchnorm/mul_1:z:0)batch_normalization_648/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
leaky_re_lu_648/LeakyRelu	LeakyRelu+batch_normalization_648/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>
dense_720/MatMul/ReadVariableOpReadVariableOp(dense_720_matmul_readvariableop_resource*
_output_shapes

:7*
dtype0
dense_720/MatMulMatMul'leaky_re_lu_648/LeakyRelu:activations:0'dense_720/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_720/BiasAdd/ReadVariableOpReadVariableOp)dense_720_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_720/BiasAddBiasAdddense_720/MatMul:product:0(dense_720/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"dense_711/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_711/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_711_matmul_readvariableop_resource*
_output_shapes

:i*
dtype0
 dense_711/kernel/Regularizer/AbsAbs7dense_711/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_711/kernel/Regularizer/SumSum$dense_711/kernel/Regularizer/Abs:y:0-dense_711/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_711/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_711/kernel/Regularizer/mulMul+dense_711/kernel/Regularizer/mul/x:output:0)dense_711/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_711/kernel/Regularizer/addAddV2+dense_711/kernel/Regularizer/Const:output:0$dense_711/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_711/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_711_matmul_readvariableop_resource*
_output_shapes

:i*
dtype0
#dense_711/kernel/Regularizer/SquareSquare:dense_711/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_711/kernel/Regularizer/Sum_1Sum'dense_711/kernel/Regularizer/Square:y:0-dense_711/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_711/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_711/kernel/Regularizer/mul_1Mul-dense_711/kernel/Regularizer/mul_1/x:output:0+dense_711/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_711/kernel/Regularizer/add_1AddV2$dense_711/kernel/Regularizer/add:z:0&dense_711/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_712/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_712/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_712_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_712/kernel/Regularizer/AbsAbs7dense_712/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_712/kernel/Regularizer/SumSum$dense_712/kernel/Regularizer/Abs:y:0-dense_712/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_712/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_712/kernel/Regularizer/mulMul+dense_712/kernel/Regularizer/mul/x:output:0)dense_712/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_712/kernel/Regularizer/addAddV2+dense_712/kernel/Regularizer/Const:output:0$dense_712/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_712/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_712_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_712/kernel/Regularizer/SquareSquare:dense_712/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_712/kernel/Regularizer/Sum_1Sum'dense_712/kernel/Regularizer/Square:y:0-dense_712/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_712/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_712/kernel/Regularizer/mul_1Mul-dense_712/kernel/Regularizer/mul_1/x:output:0+dense_712/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_712/kernel/Regularizer/add_1AddV2$dense_712/kernel/Regularizer/add:z:0&dense_712/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_713/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_713/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_713_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_713/kernel/Regularizer/AbsAbs7dense_713/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_713/kernel/Regularizer/SumSum$dense_713/kernel/Regularizer/Abs:y:0-dense_713/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_713/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_713/kernel/Regularizer/mulMul+dense_713/kernel/Regularizer/mul/x:output:0)dense_713/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_713/kernel/Regularizer/addAddV2+dense_713/kernel/Regularizer/Const:output:0$dense_713/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_713/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_713_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_713/kernel/Regularizer/SquareSquare:dense_713/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_713/kernel/Regularizer/Sum_1Sum'dense_713/kernel/Regularizer/Square:y:0-dense_713/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_713/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_713/kernel/Regularizer/mul_1Mul-dense_713/kernel/Regularizer/mul_1/x:output:0+dense_713/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_713/kernel/Regularizer/add_1AddV2$dense_713/kernel/Regularizer/add:z:0&dense_713/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_714/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_714/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_714_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_714/kernel/Regularizer/AbsAbs7dense_714/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_714/kernel/Regularizer/SumSum$dense_714/kernel/Regularizer/Abs:y:0-dense_714/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_714/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_714/kernel/Regularizer/mulMul+dense_714/kernel/Regularizer/mul/x:output:0)dense_714/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_714/kernel/Regularizer/addAddV2+dense_714/kernel/Regularizer/Const:output:0$dense_714/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_714/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_714_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_714/kernel/Regularizer/SquareSquare:dense_714/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_714/kernel/Regularizer/Sum_1Sum'dense_714/kernel/Regularizer/Square:y:0-dense_714/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_714/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_714/kernel/Regularizer/mul_1Mul-dense_714/kernel/Regularizer/mul_1/x:output:0+dense_714/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_714/kernel/Regularizer/add_1AddV2$dense_714/kernel/Regularizer/add:z:0&dense_714/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_715/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_715/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_715_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_715/kernel/Regularizer/AbsAbs7dense_715/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_715/kernel/Regularizer/SumSum$dense_715/kernel/Regularizer/Abs:y:0-dense_715/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_715/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_715/kernel/Regularizer/mulMul+dense_715/kernel/Regularizer/mul/x:output:0)dense_715/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_715/kernel/Regularizer/addAddV2+dense_715/kernel/Regularizer/Const:output:0$dense_715/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_715/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_715_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_715/kernel/Regularizer/SquareSquare:dense_715/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_715/kernel/Regularizer/Sum_1Sum'dense_715/kernel/Regularizer/Square:y:0-dense_715/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_715/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_715/kernel/Regularizer/mul_1Mul-dense_715/kernel/Regularizer/mul_1/x:output:0+dense_715/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_715/kernel/Regularizer/add_1AddV2$dense_715/kernel/Regularizer/add:z:0&dense_715/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_716/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_716/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_716_matmul_readvariableop_resource*
_output_shapes

:i=*
dtype0
 dense_716/kernel/Regularizer/AbsAbs7dense_716/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_716/kernel/Regularizer/SumSum$dense_716/kernel/Regularizer/Abs:y:0-dense_716/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_716/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_716/kernel/Regularizer/mulMul+dense_716/kernel/Regularizer/mul/x:output:0)dense_716/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_716/kernel/Regularizer/addAddV2+dense_716/kernel/Regularizer/Const:output:0$dense_716/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_716/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_716_matmul_readvariableop_resource*
_output_shapes

:i=*
dtype0
#dense_716/kernel/Regularizer/SquareSquare:dense_716/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_716/kernel/Regularizer/Sum_1Sum'dense_716/kernel/Regularizer/Square:y:0-dense_716/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_716/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_716/kernel/Regularizer/mul_1Mul-dense_716/kernel/Regularizer/mul_1/x:output:0+dense_716/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_716/kernel/Regularizer/add_1AddV2$dense_716/kernel/Regularizer/add:z:0&dense_716/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_717/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_717/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_717_matmul_readvariableop_resource*
_output_shapes

:==*
dtype0
 dense_717/kernel/Regularizer/AbsAbs7dense_717/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_717/kernel/Regularizer/SumSum$dense_717/kernel/Regularizer/Abs:y:0-dense_717/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_717/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_717/kernel/Regularizer/mulMul+dense_717/kernel/Regularizer/mul/x:output:0)dense_717/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_717/kernel/Regularizer/addAddV2+dense_717/kernel/Regularizer/Const:output:0$dense_717/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_717/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_717_matmul_readvariableop_resource*
_output_shapes

:==*
dtype0
#dense_717/kernel/Regularizer/SquareSquare:dense_717/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_717/kernel/Regularizer/Sum_1Sum'dense_717/kernel/Regularizer/Square:y:0-dense_717/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_717/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_717/kernel/Regularizer/mul_1Mul-dense_717/kernel/Regularizer/mul_1/x:output:0+dense_717/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_717/kernel/Regularizer/add_1AddV2$dense_717/kernel/Regularizer/add:z:0&dense_717/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_718/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_718/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_718_matmul_readvariableop_resource*
_output_shapes

:=7*
dtype0
 dense_718/kernel/Regularizer/AbsAbs7dense_718/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_718/kernel/Regularizer/SumSum$dense_718/kernel/Regularizer/Abs:y:0-dense_718/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_718/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_718/kernel/Regularizer/mulMul+dense_718/kernel/Regularizer/mul/x:output:0)dense_718/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_718/kernel/Regularizer/addAddV2+dense_718/kernel/Regularizer/Const:output:0$dense_718/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_718/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_718_matmul_readvariableop_resource*
_output_shapes

:=7*
dtype0
#dense_718/kernel/Regularizer/SquareSquare:dense_718/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_718/kernel/Regularizer/Sum_1Sum'dense_718/kernel/Regularizer/Square:y:0-dense_718/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_718/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_718/kernel/Regularizer/mul_1Mul-dense_718/kernel/Regularizer/mul_1/x:output:0+dense_718/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_718/kernel/Regularizer/add_1AddV2$dense_718/kernel/Regularizer/add:z:0&dense_718/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_719/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_719/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_719_matmul_readvariableop_resource*
_output_shapes

:77*
dtype0
 dense_719/kernel/Regularizer/AbsAbs7dense_719/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_719/kernel/Regularizer/SumSum$dense_719/kernel/Regularizer/Abs:y:0-dense_719/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_719/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_719/kernel/Regularizer/mulMul+dense_719/kernel/Regularizer/mul/x:output:0)dense_719/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_719/kernel/Regularizer/addAddV2+dense_719/kernel/Regularizer/Const:output:0$dense_719/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_719/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_719_matmul_readvariableop_resource*
_output_shapes

:77*
dtype0
#dense_719/kernel/Regularizer/SquareSquare:dense_719/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_719/kernel/Regularizer/Sum_1Sum'dense_719/kernel/Regularizer/Square:y:0-dense_719/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_719/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_719/kernel/Regularizer/mul_1Mul-dense_719/kernel/Regularizer/mul_1/x:output:0+dense_719/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_719/kernel/Regularizer/add_1AddV2$dense_719/kernel/Regularizer/add:z:0&dense_719/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentitydense_720/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp1^batch_normalization_640/batchnorm/ReadVariableOp3^batch_normalization_640/batchnorm/ReadVariableOp_13^batch_normalization_640/batchnorm/ReadVariableOp_25^batch_normalization_640/batchnorm/mul/ReadVariableOp1^batch_normalization_641/batchnorm/ReadVariableOp3^batch_normalization_641/batchnorm/ReadVariableOp_13^batch_normalization_641/batchnorm/ReadVariableOp_25^batch_normalization_641/batchnorm/mul/ReadVariableOp1^batch_normalization_642/batchnorm/ReadVariableOp3^batch_normalization_642/batchnorm/ReadVariableOp_13^batch_normalization_642/batchnorm/ReadVariableOp_25^batch_normalization_642/batchnorm/mul/ReadVariableOp1^batch_normalization_643/batchnorm/ReadVariableOp3^batch_normalization_643/batchnorm/ReadVariableOp_13^batch_normalization_643/batchnorm/ReadVariableOp_25^batch_normalization_643/batchnorm/mul/ReadVariableOp1^batch_normalization_644/batchnorm/ReadVariableOp3^batch_normalization_644/batchnorm/ReadVariableOp_13^batch_normalization_644/batchnorm/ReadVariableOp_25^batch_normalization_644/batchnorm/mul/ReadVariableOp1^batch_normalization_645/batchnorm/ReadVariableOp3^batch_normalization_645/batchnorm/ReadVariableOp_13^batch_normalization_645/batchnorm/ReadVariableOp_25^batch_normalization_645/batchnorm/mul/ReadVariableOp1^batch_normalization_646/batchnorm/ReadVariableOp3^batch_normalization_646/batchnorm/ReadVariableOp_13^batch_normalization_646/batchnorm/ReadVariableOp_25^batch_normalization_646/batchnorm/mul/ReadVariableOp1^batch_normalization_647/batchnorm/ReadVariableOp3^batch_normalization_647/batchnorm/ReadVariableOp_13^batch_normalization_647/batchnorm/ReadVariableOp_25^batch_normalization_647/batchnorm/mul/ReadVariableOp1^batch_normalization_648/batchnorm/ReadVariableOp3^batch_normalization_648/batchnorm/ReadVariableOp_13^batch_normalization_648/batchnorm/ReadVariableOp_25^batch_normalization_648/batchnorm/mul/ReadVariableOp!^dense_711/BiasAdd/ReadVariableOp ^dense_711/MatMul/ReadVariableOp0^dense_711/kernel/Regularizer/Abs/ReadVariableOp3^dense_711/kernel/Regularizer/Square/ReadVariableOp!^dense_712/BiasAdd/ReadVariableOp ^dense_712/MatMul/ReadVariableOp0^dense_712/kernel/Regularizer/Abs/ReadVariableOp3^dense_712/kernel/Regularizer/Square/ReadVariableOp!^dense_713/BiasAdd/ReadVariableOp ^dense_713/MatMul/ReadVariableOp0^dense_713/kernel/Regularizer/Abs/ReadVariableOp3^dense_713/kernel/Regularizer/Square/ReadVariableOp!^dense_714/BiasAdd/ReadVariableOp ^dense_714/MatMul/ReadVariableOp0^dense_714/kernel/Regularizer/Abs/ReadVariableOp3^dense_714/kernel/Regularizer/Square/ReadVariableOp!^dense_715/BiasAdd/ReadVariableOp ^dense_715/MatMul/ReadVariableOp0^dense_715/kernel/Regularizer/Abs/ReadVariableOp3^dense_715/kernel/Regularizer/Square/ReadVariableOp!^dense_716/BiasAdd/ReadVariableOp ^dense_716/MatMul/ReadVariableOp0^dense_716/kernel/Regularizer/Abs/ReadVariableOp3^dense_716/kernel/Regularizer/Square/ReadVariableOp!^dense_717/BiasAdd/ReadVariableOp ^dense_717/MatMul/ReadVariableOp0^dense_717/kernel/Regularizer/Abs/ReadVariableOp3^dense_717/kernel/Regularizer/Square/ReadVariableOp!^dense_718/BiasAdd/ReadVariableOp ^dense_718/MatMul/ReadVariableOp0^dense_718/kernel/Regularizer/Abs/ReadVariableOp3^dense_718/kernel/Regularizer/Square/ReadVariableOp!^dense_719/BiasAdd/ReadVariableOp ^dense_719/MatMul/ReadVariableOp0^dense_719/kernel/Regularizer/Abs/ReadVariableOp3^dense_719/kernel/Regularizer/Square/ReadVariableOp!^dense_720/BiasAdd/ReadVariableOp ^dense_720/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0batch_normalization_640/batchnorm/ReadVariableOp0batch_normalization_640/batchnorm/ReadVariableOp2h
2batch_normalization_640/batchnorm/ReadVariableOp_12batch_normalization_640/batchnorm/ReadVariableOp_12h
2batch_normalization_640/batchnorm/ReadVariableOp_22batch_normalization_640/batchnorm/ReadVariableOp_22l
4batch_normalization_640/batchnorm/mul/ReadVariableOp4batch_normalization_640/batchnorm/mul/ReadVariableOp2d
0batch_normalization_641/batchnorm/ReadVariableOp0batch_normalization_641/batchnorm/ReadVariableOp2h
2batch_normalization_641/batchnorm/ReadVariableOp_12batch_normalization_641/batchnorm/ReadVariableOp_12h
2batch_normalization_641/batchnorm/ReadVariableOp_22batch_normalization_641/batchnorm/ReadVariableOp_22l
4batch_normalization_641/batchnorm/mul/ReadVariableOp4batch_normalization_641/batchnorm/mul/ReadVariableOp2d
0batch_normalization_642/batchnorm/ReadVariableOp0batch_normalization_642/batchnorm/ReadVariableOp2h
2batch_normalization_642/batchnorm/ReadVariableOp_12batch_normalization_642/batchnorm/ReadVariableOp_12h
2batch_normalization_642/batchnorm/ReadVariableOp_22batch_normalization_642/batchnorm/ReadVariableOp_22l
4batch_normalization_642/batchnorm/mul/ReadVariableOp4batch_normalization_642/batchnorm/mul/ReadVariableOp2d
0batch_normalization_643/batchnorm/ReadVariableOp0batch_normalization_643/batchnorm/ReadVariableOp2h
2batch_normalization_643/batchnorm/ReadVariableOp_12batch_normalization_643/batchnorm/ReadVariableOp_12h
2batch_normalization_643/batchnorm/ReadVariableOp_22batch_normalization_643/batchnorm/ReadVariableOp_22l
4batch_normalization_643/batchnorm/mul/ReadVariableOp4batch_normalization_643/batchnorm/mul/ReadVariableOp2d
0batch_normalization_644/batchnorm/ReadVariableOp0batch_normalization_644/batchnorm/ReadVariableOp2h
2batch_normalization_644/batchnorm/ReadVariableOp_12batch_normalization_644/batchnorm/ReadVariableOp_12h
2batch_normalization_644/batchnorm/ReadVariableOp_22batch_normalization_644/batchnorm/ReadVariableOp_22l
4batch_normalization_644/batchnorm/mul/ReadVariableOp4batch_normalization_644/batchnorm/mul/ReadVariableOp2d
0batch_normalization_645/batchnorm/ReadVariableOp0batch_normalization_645/batchnorm/ReadVariableOp2h
2batch_normalization_645/batchnorm/ReadVariableOp_12batch_normalization_645/batchnorm/ReadVariableOp_12h
2batch_normalization_645/batchnorm/ReadVariableOp_22batch_normalization_645/batchnorm/ReadVariableOp_22l
4batch_normalization_645/batchnorm/mul/ReadVariableOp4batch_normalization_645/batchnorm/mul/ReadVariableOp2d
0batch_normalization_646/batchnorm/ReadVariableOp0batch_normalization_646/batchnorm/ReadVariableOp2h
2batch_normalization_646/batchnorm/ReadVariableOp_12batch_normalization_646/batchnorm/ReadVariableOp_12h
2batch_normalization_646/batchnorm/ReadVariableOp_22batch_normalization_646/batchnorm/ReadVariableOp_22l
4batch_normalization_646/batchnorm/mul/ReadVariableOp4batch_normalization_646/batchnorm/mul/ReadVariableOp2d
0batch_normalization_647/batchnorm/ReadVariableOp0batch_normalization_647/batchnorm/ReadVariableOp2h
2batch_normalization_647/batchnorm/ReadVariableOp_12batch_normalization_647/batchnorm/ReadVariableOp_12h
2batch_normalization_647/batchnorm/ReadVariableOp_22batch_normalization_647/batchnorm/ReadVariableOp_22l
4batch_normalization_647/batchnorm/mul/ReadVariableOp4batch_normalization_647/batchnorm/mul/ReadVariableOp2d
0batch_normalization_648/batchnorm/ReadVariableOp0batch_normalization_648/batchnorm/ReadVariableOp2h
2batch_normalization_648/batchnorm/ReadVariableOp_12batch_normalization_648/batchnorm/ReadVariableOp_12h
2batch_normalization_648/batchnorm/ReadVariableOp_22batch_normalization_648/batchnorm/ReadVariableOp_22l
4batch_normalization_648/batchnorm/mul/ReadVariableOp4batch_normalization_648/batchnorm/mul/ReadVariableOp2D
 dense_711/BiasAdd/ReadVariableOp dense_711/BiasAdd/ReadVariableOp2B
dense_711/MatMul/ReadVariableOpdense_711/MatMul/ReadVariableOp2b
/dense_711/kernel/Regularizer/Abs/ReadVariableOp/dense_711/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_711/kernel/Regularizer/Square/ReadVariableOp2dense_711/kernel/Regularizer/Square/ReadVariableOp2D
 dense_712/BiasAdd/ReadVariableOp dense_712/BiasAdd/ReadVariableOp2B
dense_712/MatMul/ReadVariableOpdense_712/MatMul/ReadVariableOp2b
/dense_712/kernel/Regularizer/Abs/ReadVariableOp/dense_712/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_712/kernel/Regularizer/Square/ReadVariableOp2dense_712/kernel/Regularizer/Square/ReadVariableOp2D
 dense_713/BiasAdd/ReadVariableOp dense_713/BiasAdd/ReadVariableOp2B
dense_713/MatMul/ReadVariableOpdense_713/MatMul/ReadVariableOp2b
/dense_713/kernel/Regularizer/Abs/ReadVariableOp/dense_713/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_713/kernel/Regularizer/Square/ReadVariableOp2dense_713/kernel/Regularizer/Square/ReadVariableOp2D
 dense_714/BiasAdd/ReadVariableOp dense_714/BiasAdd/ReadVariableOp2B
dense_714/MatMul/ReadVariableOpdense_714/MatMul/ReadVariableOp2b
/dense_714/kernel/Regularizer/Abs/ReadVariableOp/dense_714/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_714/kernel/Regularizer/Square/ReadVariableOp2dense_714/kernel/Regularizer/Square/ReadVariableOp2D
 dense_715/BiasAdd/ReadVariableOp dense_715/BiasAdd/ReadVariableOp2B
dense_715/MatMul/ReadVariableOpdense_715/MatMul/ReadVariableOp2b
/dense_715/kernel/Regularizer/Abs/ReadVariableOp/dense_715/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_715/kernel/Regularizer/Square/ReadVariableOp2dense_715/kernel/Regularizer/Square/ReadVariableOp2D
 dense_716/BiasAdd/ReadVariableOp dense_716/BiasAdd/ReadVariableOp2B
dense_716/MatMul/ReadVariableOpdense_716/MatMul/ReadVariableOp2b
/dense_716/kernel/Regularizer/Abs/ReadVariableOp/dense_716/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_716/kernel/Regularizer/Square/ReadVariableOp2dense_716/kernel/Regularizer/Square/ReadVariableOp2D
 dense_717/BiasAdd/ReadVariableOp dense_717/BiasAdd/ReadVariableOp2B
dense_717/MatMul/ReadVariableOpdense_717/MatMul/ReadVariableOp2b
/dense_717/kernel/Regularizer/Abs/ReadVariableOp/dense_717/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_717/kernel/Regularizer/Square/ReadVariableOp2dense_717/kernel/Regularizer/Square/ReadVariableOp2D
 dense_718/BiasAdd/ReadVariableOp dense_718/BiasAdd/ReadVariableOp2B
dense_718/MatMul/ReadVariableOpdense_718/MatMul/ReadVariableOp2b
/dense_718/kernel/Regularizer/Abs/ReadVariableOp/dense_718/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_718/kernel/Regularizer/Square/ReadVariableOp2dense_718/kernel/Regularizer/Square/ReadVariableOp2D
 dense_719/BiasAdd/ReadVariableOp dense_719/BiasAdd/ReadVariableOp2B
dense_719/MatMul/ReadVariableOpdense_719/MatMul/ReadVariableOp2b
/dense_719/kernel/Regularizer/Abs/ReadVariableOp/dense_719/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_719/kernel/Regularizer/Square/ReadVariableOp2dense_719/kernel/Regularizer/Square/ReadVariableOp2D
 dense_720/BiasAdd/ReadVariableOp dense_720/BiasAdd/ReadVariableOp2B
dense_720/MatMul/ReadVariableOpdense_720/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
%
í
T__inference_batch_normalization_640_layer_call_and_return_conditional_losses_1125106

inputs5
'assignmovingavg_readvariableop_resource:i7
)assignmovingavg_1_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i/
!batchnorm_readvariableop_resource:i
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:i
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿil
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:i*
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
:i*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ix
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i¬
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
:i*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:i~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i´
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿih
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:iv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_644_layer_call_and_return_conditional_losses_1125662

inputs5
'assignmovingavg_readvariableop_resource:i7
)assignmovingavg_1_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i/
!batchnorm_readvariableop_resource:i
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:i
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿil
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:i*
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
:i*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ix
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i¬
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
:i*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:i~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i´
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿih
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:iv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_641_layer_call_and_return_conditional_losses_1121605

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿi:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_647_layer_call_fn_1126012

inputs
unknown:7
	unknown_0:7
	unknown_1:7
	unknown_2:7
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_647_layer_call_and_return_conditional_losses_1121359o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_646_layer_call_fn_1125945

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
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_646_layer_call_and_return_conditional_losses_1121840`
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
×
Ù
%__inference_signature_wrapper_1124930
normalization_71_input
unknown
	unknown_0
	unknown_1:i
	unknown_2:i
	unknown_3:i
	unknown_4:i
	unknown_5:i
	unknown_6:i
	unknown_7:ii
	unknown_8:i
	unknown_9:i

unknown_10:i

unknown_11:i

unknown_12:i

unknown_13:ii

unknown_14:i

unknown_15:i

unknown_16:i

unknown_17:i

unknown_18:i

unknown_19:ii

unknown_20:i

unknown_21:i

unknown_22:i

unknown_23:i

unknown_24:i

unknown_25:ii

unknown_26:i

unknown_27:i

unknown_28:i

unknown_29:i

unknown_30:i

unknown_31:i=

unknown_32:=

unknown_33:=

unknown_34:=

unknown_35:=

unknown_36:=

unknown_37:==

unknown_38:=

unknown_39:=

unknown_40:=

unknown_41:=

unknown_42:=

unknown_43:=7

unknown_44:7

unknown_45:7

unknown_46:7

unknown_47:7

unknown_48:7

unknown_49:77

unknown_50:7

unknown_51:7

unknown_52:7

unknown_53:7

unknown_54:7

unknown_55:7

unknown_56:
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallnormalization_71_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1120761o
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
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_71_input:$ 

_output_shapes

::$ 

_output_shapes

:
%
í
T__inference_batch_normalization_641_layer_call_and_return_conditional_losses_1125245

inputs5
'assignmovingavg_readvariableop_resource:i7
)assignmovingavg_1_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i/
!batchnorm_readvariableop_resource:i
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:i
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿil
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:i*
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
:i*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ix
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i¬
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
:i*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:i~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i´
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿih
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:iv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_645_layer_call_and_return_conditional_losses_1121242

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
æ
h
L__inference_leaky_re_lu_645_layer_call_and_return_conditional_losses_1121793

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
%
í
T__inference_batch_normalization_640_layer_call_and_return_conditional_losses_1120832

inputs5
'assignmovingavg_readvariableop_resource:i7
)assignmovingavg_1_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i/
!batchnorm_readvariableop_resource:i
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:i
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿil
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:i*
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
:i*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ix
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i¬
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
:i*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:i~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i´
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿih
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:iv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_646_layer_call_and_return_conditional_losses_1121840

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
­
M
1__inference_leaky_re_lu_640_layer_call_fn_1125111

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
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_640_layer_call_and_return_conditional_losses_1121558`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿi:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
Ü'
Ó
__inference_adapt_step_1124977
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
%
í
T__inference_batch_normalization_646_layer_call_and_return_conditional_losses_1121324

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
®
Ô
9__inference_batch_normalization_645_layer_call_fn_1125734

inputs
unknown:=
	unknown_0:=
	unknown_1:=
	unknown_2:=
identity¢StatefulPartitionedCall
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_645_layer_call_and_return_conditional_losses_1121195o
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
­
M
1__inference_leaky_re_lu_641_layer_call_fn_1125250

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
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_641_layer_call_and_return_conditional_losses_1121605`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿi:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_644_layer_call_and_return_conditional_losses_1121160

inputs5
'assignmovingavg_readvariableop_resource:i7
)assignmovingavg_1_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i/
!batchnorm_readvariableop_resource:i
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:i
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿil
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:i*
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
:i*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ix
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i¬
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
:i*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:i~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i´
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿih
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:iv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_646_layer_call_and_return_conditional_losses_1125950

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
Ñ
³
T__inference_batch_normalization_641_layer_call_and_return_conditional_losses_1120867

inputs/
!batchnorm_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i1
#batchnorm_readvariableop_1_resource:i1
#batchnorm_readvariableop_2_resource:i
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:iz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_642_layer_call_and_return_conditional_losses_1120949

inputs/
!batchnorm_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i1
#batchnorm_readvariableop_1_resource:i1
#batchnorm_readvariableop_2_resource:i
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:iz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
ã
B
 __inference__traced_save_1126875
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	/
+savev2_dense_711_kernel_read_readvariableop-
)savev2_dense_711_bias_read_readvariableop<
8savev2_batch_normalization_640_gamma_read_readvariableop;
7savev2_batch_normalization_640_beta_read_readvariableopB
>savev2_batch_normalization_640_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_640_moving_variance_read_readvariableop/
+savev2_dense_712_kernel_read_readvariableop-
)savev2_dense_712_bias_read_readvariableop<
8savev2_batch_normalization_641_gamma_read_readvariableop;
7savev2_batch_normalization_641_beta_read_readvariableopB
>savev2_batch_normalization_641_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_641_moving_variance_read_readvariableop/
+savev2_dense_713_kernel_read_readvariableop-
)savev2_dense_713_bias_read_readvariableop<
8savev2_batch_normalization_642_gamma_read_readvariableop;
7savev2_batch_normalization_642_beta_read_readvariableopB
>savev2_batch_normalization_642_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_642_moving_variance_read_readvariableop/
+savev2_dense_714_kernel_read_readvariableop-
)savev2_dense_714_bias_read_readvariableop<
8savev2_batch_normalization_643_gamma_read_readvariableop;
7savev2_batch_normalization_643_beta_read_readvariableopB
>savev2_batch_normalization_643_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_643_moving_variance_read_readvariableop/
+savev2_dense_715_kernel_read_readvariableop-
)savev2_dense_715_bias_read_readvariableop<
8savev2_batch_normalization_644_gamma_read_readvariableop;
7savev2_batch_normalization_644_beta_read_readvariableopB
>savev2_batch_normalization_644_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_644_moving_variance_read_readvariableop/
+savev2_dense_716_kernel_read_readvariableop-
)savev2_dense_716_bias_read_readvariableop<
8savev2_batch_normalization_645_gamma_read_readvariableop;
7savev2_batch_normalization_645_beta_read_readvariableopB
>savev2_batch_normalization_645_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_645_moving_variance_read_readvariableop/
+savev2_dense_717_kernel_read_readvariableop-
)savev2_dense_717_bias_read_readvariableop<
8savev2_batch_normalization_646_gamma_read_readvariableop;
7savev2_batch_normalization_646_beta_read_readvariableopB
>savev2_batch_normalization_646_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_646_moving_variance_read_readvariableop/
+savev2_dense_718_kernel_read_readvariableop-
)savev2_dense_718_bias_read_readvariableop<
8savev2_batch_normalization_647_gamma_read_readvariableop;
7savev2_batch_normalization_647_beta_read_readvariableopB
>savev2_batch_normalization_647_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_647_moving_variance_read_readvariableop/
+savev2_dense_719_kernel_read_readvariableop-
)savev2_dense_719_bias_read_readvariableop<
8savev2_batch_normalization_648_gamma_read_readvariableop;
7savev2_batch_normalization_648_beta_read_readvariableopB
>savev2_batch_normalization_648_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_648_moving_variance_read_readvariableop/
+savev2_dense_720_kernel_read_readvariableop-
)savev2_dense_720_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_711_kernel_m_read_readvariableop4
0savev2_adam_dense_711_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_640_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_640_beta_m_read_readvariableop6
2savev2_adam_dense_712_kernel_m_read_readvariableop4
0savev2_adam_dense_712_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_641_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_641_beta_m_read_readvariableop6
2savev2_adam_dense_713_kernel_m_read_readvariableop4
0savev2_adam_dense_713_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_642_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_642_beta_m_read_readvariableop6
2savev2_adam_dense_714_kernel_m_read_readvariableop4
0savev2_adam_dense_714_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_643_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_643_beta_m_read_readvariableop6
2savev2_adam_dense_715_kernel_m_read_readvariableop4
0savev2_adam_dense_715_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_644_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_644_beta_m_read_readvariableop6
2savev2_adam_dense_716_kernel_m_read_readvariableop4
0savev2_adam_dense_716_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_645_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_645_beta_m_read_readvariableop6
2savev2_adam_dense_717_kernel_m_read_readvariableop4
0savev2_adam_dense_717_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_646_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_646_beta_m_read_readvariableop6
2savev2_adam_dense_718_kernel_m_read_readvariableop4
0savev2_adam_dense_718_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_647_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_647_beta_m_read_readvariableop6
2savev2_adam_dense_719_kernel_m_read_readvariableop4
0savev2_adam_dense_719_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_648_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_648_beta_m_read_readvariableop6
2savev2_adam_dense_720_kernel_m_read_readvariableop4
0savev2_adam_dense_720_bias_m_read_readvariableop6
2savev2_adam_dense_711_kernel_v_read_readvariableop4
0savev2_adam_dense_711_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_640_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_640_beta_v_read_readvariableop6
2savev2_adam_dense_712_kernel_v_read_readvariableop4
0savev2_adam_dense_712_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_641_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_641_beta_v_read_readvariableop6
2savev2_adam_dense_713_kernel_v_read_readvariableop4
0savev2_adam_dense_713_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_642_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_642_beta_v_read_readvariableop6
2savev2_adam_dense_714_kernel_v_read_readvariableop4
0savev2_adam_dense_714_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_643_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_643_beta_v_read_readvariableop6
2savev2_adam_dense_715_kernel_v_read_readvariableop4
0savev2_adam_dense_715_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_644_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_644_beta_v_read_readvariableop6
2savev2_adam_dense_716_kernel_v_read_readvariableop4
0savev2_adam_dense_716_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_645_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_645_beta_v_read_readvariableop6
2savev2_adam_dense_717_kernel_v_read_readvariableop4
0savev2_adam_dense_717_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_646_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_646_beta_v_read_readvariableop6
2savev2_adam_dense_718_kernel_v_read_readvariableop4
0savev2_adam_dense_718_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_647_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_647_beta_v_read_readvariableop6
2savev2_adam_dense_719_kernel_v_read_readvariableop4
0savev2_adam_dense_719_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_648_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_648_beta_v_read_readvariableop6
2savev2_adam_dense_720_kernel_v_read_readvariableop4
0savev2_adam_dense_720_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop+savev2_dense_711_kernel_read_readvariableop)savev2_dense_711_bias_read_readvariableop8savev2_batch_normalization_640_gamma_read_readvariableop7savev2_batch_normalization_640_beta_read_readvariableop>savev2_batch_normalization_640_moving_mean_read_readvariableopBsavev2_batch_normalization_640_moving_variance_read_readvariableop+savev2_dense_712_kernel_read_readvariableop)savev2_dense_712_bias_read_readvariableop8savev2_batch_normalization_641_gamma_read_readvariableop7savev2_batch_normalization_641_beta_read_readvariableop>savev2_batch_normalization_641_moving_mean_read_readvariableopBsavev2_batch_normalization_641_moving_variance_read_readvariableop+savev2_dense_713_kernel_read_readvariableop)savev2_dense_713_bias_read_readvariableop8savev2_batch_normalization_642_gamma_read_readvariableop7savev2_batch_normalization_642_beta_read_readvariableop>savev2_batch_normalization_642_moving_mean_read_readvariableopBsavev2_batch_normalization_642_moving_variance_read_readvariableop+savev2_dense_714_kernel_read_readvariableop)savev2_dense_714_bias_read_readvariableop8savev2_batch_normalization_643_gamma_read_readvariableop7savev2_batch_normalization_643_beta_read_readvariableop>savev2_batch_normalization_643_moving_mean_read_readvariableopBsavev2_batch_normalization_643_moving_variance_read_readvariableop+savev2_dense_715_kernel_read_readvariableop)savev2_dense_715_bias_read_readvariableop8savev2_batch_normalization_644_gamma_read_readvariableop7savev2_batch_normalization_644_beta_read_readvariableop>savev2_batch_normalization_644_moving_mean_read_readvariableopBsavev2_batch_normalization_644_moving_variance_read_readvariableop+savev2_dense_716_kernel_read_readvariableop)savev2_dense_716_bias_read_readvariableop8savev2_batch_normalization_645_gamma_read_readvariableop7savev2_batch_normalization_645_beta_read_readvariableop>savev2_batch_normalization_645_moving_mean_read_readvariableopBsavev2_batch_normalization_645_moving_variance_read_readvariableop+savev2_dense_717_kernel_read_readvariableop)savev2_dense_717_bias_read_readvariableop8savev2_batch_normalization_646_gamma_read_readvariableop7savev2_batch_normalization_646_beta_read_readvariableop>savev2_batch_normalization_646_moving_mean_read_readvariableopBsavev2_batch_normalization_646_moving_variance_read_readvariableop+savev2_dense_718_kernel_read_readvariableop)savev2_dense_718_bias_read_readvariableop8savev2_batch_normalization_647_gamma_read_readvariableop7savev2_batch_normalization_647_beta_read_readvariableop>savev2_batch_normalization_647_moving_mean_read_readvariableopBsavev2_batch_normalization_647_moving_variance_read_readvariableop+savev2_dense_719_kernel_read_readvariableop)savev2_dense_719_bias_read_readvariableop8savev2_batch_normalization_648_gamma_read_readvariableop7savev2_batch_normalization_648_beta_read_readvariableop>savev2_batch_normalization_648_moving_mean_read_readvariableopBsavev2_batch_normalization_648_moving_variance_read_readvariableop+savev2_dense_720_kernel_read_readvariableop)savev2_dense_720_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_711_kernel_m_read_readvariableop0savev2_adam_dense_711_bias_m_read_readvariableop?savev2_adam_batch_normalization_640_gamma_m_read_readvariableop>savev2_adam_batch_normalization_640_beta_m_read_readvariableop2savev2_adam_dense_712_kernel_m_read_readvariableop0savev2_adam_dense_712_bias_m_read_readvariableop?savev2_adam_batch_normalization_641_gamma_m_read_readvariableop>savev2_adam_batch_normalization_641_beta_m_read_readvariableop2savev2_adam_dense_713_kernel_m_read_readvariableop0savev2_adam_dense_713_bias_m_read_readvariableop?savev2_adam_batch_normalization_642_gamma_m_read_readvariableop>savev2_adam_batch_normalization_642_beta_m_read_readvariableop2savev2_adam_dense_714_kernel_m_read_readvariableop0savev2_adam_dense_714_bias_m_read_readvariableop?savev2_adam_batch_normalization_643_gamma_m_read_readvariableop>savev2_adam_batch_normalization_643_beta_m_read_readvariableop2savev2_adam_dense_715_kernel_m_read_readvariableop0savev2_adam_dense_715_bias_m_read_readvariableop?savev2_adam_batch_normalization_644_gamma_m_read_readvariableop>savev2_adam_batch_normalization_644_beta_m_read_readvariableop2savev2_adam_dense_716_kernel_m_read_readvariableop0savev2_adam_dense_716_bias_m_read_readvariableop?savev2_adam_batch_normalization_645_gamma_m_read_readvariableop>savev2_adam_batch_normalization_645_beta_m_read_readvariableop2savev2_adam_dense_717_kernel_m_read_readvariableop0savev2_adam_dense_717_bias_m_read_readvariableop?savev2_adam_batch_normalization_646_gamma_m_read_readvariableop>savev2_adam_batch_normalization_646_beta_m_read_readvariableop2savev2_adam_dense_718_kernel_m_read_readvariableop0savev2_adam_dense_718_bias_m_read_readvariableop?savev2_adam_batch_normalization_647_gamma_m_read_readvariableop>savev2_adam_batch_normalization_647_beta_m_read_readvariableop2savev2_adam_dense_719_kernel_m_read_readvariableop0savev2_adam_dense_719_bias_m_read_readvariableop?savev2_adam_batch_normalization_648_gamma_m_read_readvariableop>savev2_adam_batch_normalization_648_beta_m_read_readvariableop2savev2_adam_dense_720_kernel_m_read_readvariableop0savev2_adam_dense_720_bias_m_read_readvariableop2savev2_adam_dense_711_kernel_v_read_readvariableop0savev2_adam_dense_711_bias_v_read_readvariableop?savev2_adam_batch_normalization_640_gamma_v_read_readvariableop>savev2_adam_batch_normalization_640_beta_v_read_readvariableop2savev2_adam_dense_712_kernel_v_read_readvariableop0savev2_adam_dense_712_bias_v_read_readvariableop?savev2_adam_batch_normalization_641_gamma_v_read_readvariableop>savev2_adam_batch_normalization_641_beta_v_read_readvariableop2savev2_adam_dense_713_kernel_v_read_readvariableop0savev2_adam_dense_713_bias_v_read_readvariableop?savev2_adam_batch_normalization_642_gamma_v_read_readvariableop>savev2_adam_batch_normalization_642_beta_v_read_readvariableop2savev2_adam_dense_714_kernel_v_read_readvariableop0savev2_adam_dense_714_bias_v_read_readvariableop?savev2_adam_batch_normalization_643_gamma_v_read_readvariableop>savev2_adam_batch_normalization_643_beta_v_read_readvariableop2savev2_adam_dense_715_kernel_v_read_readvariableop0savev2_adam_dense_715_bias_v_read_readvariableop?savev2_adam_batch_normalization_644_gamma_v_read_readvariableop>savev2_adam_batch_normalization_644_beta_v_read_readvariableop2savev2_adam_dense_716_kernel_v_read_readvariableop0savev2_adam_dense_716_bias_v_read_readvariableop?savev2_adam_batch_normalization_645_gamma_v_read_readvariableop>savev2_adam_batch_normalization_645_beta_v_read_readvariableop2savev2_adam_dense_717_kernel_v_read_readvariableop0savev2_adam_dense_717_bias_v_read_readvariableop?savev2_adam_batch_normalization_646_gamma_v_read_readvariableop>savev2_adam_batch_normalization_646_beta_v_read_readvariableop2savev2_adam_dense_718_kernel_v_read_readvariableop0savev2_adam_dense_718_bias_v_read_readvariableop?savev2_adam_batch_normalization_647_gamma_v_read_readvariableop>savev2_adam_batch_normalization_647_beta_v_read_readvariableop2savev2_adam_dense_719_kernel_v_read_readvariableop0savev2_adam_dense_719_bias_v_read_readvariableop?savev2_adam_batch_normalization_648_gamma_v_read_readvariableop>savev2_adam_batch_normalization_648_beta_v_read_readvariableop2savev2_adam_dense_720_kernel_v_read_readvariableop0savev2_adam_dense_720_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
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
®: ::: :i:i:i:i:i:i:ii:i:i:i:i:i:ii:i:i:i:i:i:ii:i:i:i:i:i:ii:i:i:i:i:i:i=:=:=:=:=:=:==:=:=:=:=:=:=7:7:7:7:7:7:77:7:7:7:7:7:7:: : : : : : :i:i:i:i:ii:i:i:i:ii:i:i:i:ii:i:i:i:ii:i:i:i:i=:=:=:=:==:=:=:=:=7:7:7:7:77:7:7:7:7::i:i:i:i:ii:i:i:i:ii:i:i:i:ii:i:i:i:ii:i:i:i:i=:=:=:=:==:=:=:=:=7:7:7:7:77:7:7:7:7:: 2(
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

:i: 

_output_shapes
:i: 

_output_shapes
:i: 

_output_shapes
:i: 

_output_shapes
:i: 	

_output_shapes
:i:$
 

_output_shapes

:ii: 

_output_shapes
:i: 

_output_shapes
:i: 

_output_shapes
:i: 

_output_shapes
:i: 

_output_shapes
:i:$ 

_output_shapes

:ii: 

_output_shapes
:i: 

_output_shapes
:i: 

_output_shapes
:i: 

_output_shapes
:i: 

_output_shapes
:i:$ 

_output_shapes

:ii: 

_output_shapes
:i: 

_output_shapes
:i: 

_output_shapes
:i: 

_output_shapes
:i: 

_output_shapes
:i:$ 

_output_shapes

:ii: 

_output_shapes
:i: 

_output_shapes
:i: 

_output_shapes
:i:  

_output_shapes
:i: !

_output_shapes
:i:$" 

_output_shapes

:i=: #

_output_shapes
:=: $

_output_shapes
:=: %

_output_shapes
:=: &

_output_shapes
:=: '

_output_shapes
:=:$( 

_output_shapes

:==: )
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

:=7: /

_output_shapes
:7: 0

_output_shapes
:7: 1

_output_shapes
:7: 2

_output_shapes
:7: 3

_output_shapes
:7:$4 

_output_shapes

:77: 5

_output_shapes
:7: 6

_output_shapes
:7: 7

_output_shapes
:7: 8

_output_shapes
:7: 9

_output_shapes
:7:$: 

_output_shapes

:7: ;
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

:i: C

_output_shapes
:i: D

_output_shapes
:i: E

_output_shapes
:i:$F 

_output_shapes

:ii: G

_output_shapes
:i: H

_output_shapes
:i: I

_output_shapes
:i:$J 

_output_shapes

:ii: K

_output_shapes
:i: L

_output_shapes
:i: M

_output_shapes
:i:$N 

_output_shapes

:ii: O

_output_shapes
:i: P

_output_shapes
:i: Q

_output_shapes
:i:$R 

_output_shapes

:ii: S

_output_shapes
:i: T

_output_shapes
:i: U

_output_shapes
:i:$V 

_output_shapes

:i=: W

_output_shapes
:=: X

_output_shapes
:=: Y

_output_shapes
:=:$Z 

_output_shapes

:==: [
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

:=7: _

_output_shapes
:7: `

_output_shapes
:7: a

_output_shapes
:7:$b 

_output_shapes

:77: c

_output_shapes
:7: d

_output_shapes
:7: e

_output_shapes
:7:$f 

_output_shapes

:7: g

_output_shapes
::$h 

_output_shapes

:i: i

_output_shapes
:i: j

_output_shapes
:i: k

_output_shapes
:i:$l 

_output_shapes

:ii: m

_output_shapes
:i: n

_output_shapes
:i: o

_output_shapes
:i:$p 

_output_shapes

:ii: q

_output_shapes
:i: r

_output_shapes
:i: s

_output_shapes
:i:$t 

_output_shapes

:ii: u

_output_shapes
:i: v

_output_shapes
:i: w

_output_shapes
:i:$x 

_output_shapes

:ii: y

_output_shapes
:i: z

_output_shapes
:i: {

_output_shapes
:i:$| 

_output_shapes

:i=: }

_output_shapes
:=: ~

_output_shapes
:=: 

_output_shapes
:=:% 

_output_shapes

:==:!
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

:=7:!

_output_shapes
:7:!

_output_shapes
:7:!

_output_shapes
:7:% 

_output_shapes

:77:!

_output_shapes
:7:!

_output_shapes
:7:!

_output_shapes
:7:% 

_output_shapes

:7:!

_output_shapes
::

_output_shapes
: 
Ñ
³
T__inference_batch_normalization_647_layer_call_and_return_conditional_losses_1126045

inputs/
!batchnorm_readvariableop_resource:73
%batchnorm_mul_readvariableop_resource:71
#batchnorm_readvariableop_1_resource:71
#batchnorm_readvariableop_2_resource:7
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:7*
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
:7P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:7~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:7*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:7z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:7*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:7r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
¥
Þ
F__inference_dense_711_layer_call_and_return_conditional_losses_1121538

inputs0
matmul_readvariableop_resource:i-
biasadd_readvariableop_resource:i
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_711/kernel/Regularizer/Abs/ReadVariableOp¢2dense_711/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:i*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿir
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:i*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿig
"dense_711/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_711/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:i*
dtype0
 dense_711/kernel/Regularizer/AbsAbs7dense_711/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_711/kernel/Regularizer/SumSum$dense_711/kernel/Regularizer/Abs:y:0-dense_711/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_711/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_711/kernel/Regularizer/mulMul+dense_711/kernel/Regularizer/mul/x:output:0)dense_711/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_711/kernel/Regularizer/addAddV2+dense_711/kernel/Regularizer/Const:output:0$dense_711/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_711/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:i*
dtype0
#dense_711/kernel/Regularizer/SquareSquare:dense_711/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_711/kernel/Regularizer/Sum_1Sum'dense_711/kernel/Regularizer/Square:y:0-dense_711/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_711/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_711/kernel/Regularizer/mul_1Mul-dense_711/kernel/Regularizer/mul_1/x:output:0+dense_711/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_711/kernel/Regularizer/add_1AddV2$dense_711/kernel/Regularizer/add:z:0&dense_711/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÞ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_711/kernel/Regularizer/Abs/ReadVariableOp3^dense_711/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_711/kernel/Regularizer/Abs/ReadVariableOp/dense_711/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_711/kernel/Regularizer/Square/ReadVariableOp2dense_711/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_648_layer_call_and_return_conditional_losses_1126184

inputs/
!batchnorm_readvariableop_resource:73
%batchnorm_mul_readvariableop_resource:71
#batchnorm_readvariableop_1_resource:71
#batchnorm_readvariableop_2_resource:7
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:7*
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
:7P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:7~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:7*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:7z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:7*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:7r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
¥
Þ
F__inference_dense_713_layer_call_and_return_conditional_losses_1121632

inputs0
matmul_readvariableop_resource:ii-
biasadd_readvariableop_resource:i
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_713/kernel/Regularizer/Abs/ReadVariableOp¢2dense_713/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿir
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:i*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿig
"dense_713/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_713/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_713/kernel/Regularizer/AbsAbs7dense_713/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_713/kernel/Regularizer/SumSum$dense_713/kernel/Regularizer/Abs:y:0-dense_713/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_713/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_713/kernel/Regularizer/mulMul+dense_713/kernel/Regularizer/mul/x:output:0)dense_713/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_713/kernel/Regularizer/addAddV2+dense_713/kernel/Regularizer/Const:output:0$dense_713/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_713/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_713/kernel/Regularizer/SquareSquare:dense_713/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_713/kernel/Regularizer/Sum_1Sum'dense_713/kernel/Regularizer/Square:y:0-dense_713/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_713/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_713/kernel/Regularizer/mul_1Mul-dense_713/kernel/Regularizer/mul_1/x:output:0+dense_713/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_713/kernel/Regularizer/add_1AddV2$dense_713/kernel/Regularizer/add:z:0&dense_713/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÞ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_713/kernel/Regularizer/Abs/ReadVariableOp3^dense_713/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_713/kernel/Regularizer/Abs/ReadVariableOp/dense_713/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_713/kernel/Regularizer/Square/ReadVariableOp2dense_713/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
¥
Þ
F__inference_dense_713_layer_call_and_return_conditional_losses_1125304

inputs0
matmul_readvariableop_resource:ii-
biasadd_readvariableop_resource:i
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_713/kernel/Regularizer/Abs/ReadVariableOp¢2dense_713/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿir
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:i*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿig
"dense_713/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_713/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_713/kernel/Regularizer/AbsAbs7dense_713/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_713/kernel/Regularizer/SumSum$dense_713/kernel/Regularizer/Abs:y:0-dense_713/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_713/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_713/kernel/Regularizer/mulMul+dense_713/kernel/Regularizer/mul/x:output:0)dense_713/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_713/kernel/Regularizer/addAddV2+dense_713/kernel/Regularizer/Const:output:0$dense_713/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_713/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_713/kernel/Regularizer/SquareSquare:dense_713/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_713/kernel/Regularizer/Sum_1Sum'dense_713/kernel/Regularizer/Square:y:0-dense_713/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_713/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_713/kernel/Regularizer/mul_1Mul-dense_713/kernel/Regularizer/mul_1/x:output:0+dense_713/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_713/kernel/Regularizer/add_1AddV2$dense_713/kernel/Regularizer/add:z:0&dense_713/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÞ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_713/kernel/Regularizer/Abs/ReadVariableOp3^dense_713/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_713/kernel/Regularizer/Abs/ReadVariableOp/dense_713/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_713/kernel/Regularizer/Square/ReadVariableOp2dense_713/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_645_layer_call_and_return_conditional_losses_1125811

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
¥
Þ
F__inference_dense_719_layer_call_and_return_conditional_losses_1126138

inputs0
matmul_readvariableop_resource:77-
biasadd_readvariableop_resource:7
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_719/kernel/Regularizer/Abs/ReadVariableOp¢2dense_719/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:77*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:7*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7g
"dense_719/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_719/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:77*
dtype0
 dense_719/kernel/Regularizer/AbsAbs7dense_719/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_719/kernel/Regularizer/SumSum$dense_719/kernel/Regularizer/Abs:y:0-dense_719/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_719/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_719/kernel/Regularizer/mulMul+dense_719/kernel/Regularizer/mul/x:output:0)dense_719/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_719/kernel/Regularizer/addAddV2+dense_719/kernel/Regularizer/Const:output:0$dense_719/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_719/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:77*
dtype0
#dense_719/kernel/Regularizer/SquareSquare:dense_719/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_719/kernel/Regularizer/Sum_1Sum'dense_719/kernel/Regularizer/Square:y:0-dense_719/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_719/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_719/kernel/Regularizer/mul_1Mul-dense_719/kernel/Regularizer/mul_1/x:output:0+dense_719/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_719/kernel/Regularizer/add_1AddV2$dense_719/kernel/Regularizer/add:z:0&dense_719/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7Þ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_719/kernel/Regularizer/Abs/ReadVariableOp3^dense_719/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_719/kernel/Regularizer/Abs/ReadVariableOp/dense_719/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_719/kernel/Regularizer/Square/ReadVariableOp2dense_719/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_645_layer_call_fn_1125747

inputs
unknown:=
	unknown_0:=
	unknown_1:=
	unknown_2:=
identity¢StatefulPartitionedCall
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_645_layer_call_and_return_conditional_losses_1121242o
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
¥
Þ
F__inference_dense_716_layer_call_and_return_conditional_losses_1125721

inputs0
matmul_readvariableop_resource:i=-
biasadd_readvariableop_resource:=
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_716/kernel/Regularizer/Abs/ReadVariableOp¢2dense_716/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:i=*
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
:ÿÿÿÿÿÿÿÿÿ=g
"dense_716/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_716/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:i=*
dtype0
 dense_716/kernel/Regularizer/AbsAbs7dense_716/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_716/kernel/Regularizer/SumSum$dense_716/kernel/Regularizer/Abs:y:0-dense_716/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_716/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_716/kernel/Regularizer/mulMul+dense_716/kernel/Regularizer/mul/x:output:0)dense_716/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_716/kernel/Regularizer/addAddV2+dense_716/kernel/Regularizer/Const:output:0$dense_716/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_716/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:i=*
dtype0
#dense_716/kernel/Regularizer/SquareSquare:dense_716/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_716/kernel/Regularizer/Sum_1Sum'dense_716/kernel/Regularizer/Square:y:0-dense_716/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_716/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_716/kernel/Regularizer/mul_1Mul-dense_716/kernel/Regularizer/mul_1/x:output:0+dense_716/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_716/kernel/Regularizer/add_1AddV2$dense_716/kernel/Regularizer/add:z:0&dense_716/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=Þ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_716/kernel/Regularizer/Abs/ReadVariableOp3^dense_716/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_716/kernel/Regularizer/Abs/ReadVariableOp/dense_716/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_716/kernel/Regularizer/Square/ReadVariableOp2dense_716/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
¥
Þ
F__inference_dense_715_layer_call_and_return_conditional_losses_1121726

inputs0
matmul_readvariableop_resource:ii-
biasadd_readvariableop_resource:i
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_715/kernel/Regularizer/Abs/ReadVariableOp¢2dense_715/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿir
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:i*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿig
"dense_715/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_715/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_715/kernel/Regularizer/AbsAbs7dense_715/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_715/kernel/Regularizer/SumSum$dense_715/kernel/Regularizer/Abs:y:0-dense_715/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_715/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_715/kernel/Regularizer/mulMul+dense_715/kernel/Regularizer/mul/x:output:0)dense_715/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_715/kernel/Regularizer/addAddV2+dense_715/kernel/Regularizer/Const:output:0$dense_715/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_715/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_715/kernel/Regularizer/SquareSquare:dense_715/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_715/kernel/Regularizer/Sum_1Sum'dense_715/kernel/Regularizer/Square:y:0-dense_715/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_715/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_715/kernel/Regularizer/mul_1Mul-dense_715/kernel/Regularizer/mul_1/x:output:0+dense_715/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_715/kernel/Regularizer/add_1AddV2$dense_715/kernel/Regularizer/add:z:0&dense_715/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÞ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_715/kernel/Regularizer/Abs/ReadVariableOp3^dense_715/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_715/kernel/Regularizer/Abs/ReadVariableOp/dense_715/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_715/kernel/Regularizer/Square/ReadVariableOp2dense_715/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_648_layer_call_and_return_conditional_losses_1126218

inputs5
'assignmovingavg_readvariableop_resource:77
)assignmovingavg_1_readvariableop_resource:73
%batchnorm_mul_readvariableop_resource:7/
!batchnorm_readvariableop_resource:7
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:7
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:7*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:7*
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
:7*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:7x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:7¬
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
:7*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:7~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:7´
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
:7P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:7~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:7v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:7r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_648_layer_call_and_return_conditional_losses_1121488

inputs5
'assignmovingavg_readvariableop_resource:77
)assignmovingavg_1_readvariableop_resource:73
%batchnorm_mul_readvariableop_resource:7/
!batchnorm_readvariableop_resource:7
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:7
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:7*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:7*
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
:7*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:7x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:7¬
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
:7*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:7~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:7´
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
:7P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:7~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:7v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:7r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
¥
Þ
F__inference_dense_712_layer_call_and_return_conditional_losses_1125165

inputs0
matmul_readvariableop_resource:ii-
biasadd_readvariableop_resource:i
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_712/kernel/Regularizer/Abs/ReadVariableOp¢2dense_712/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿir
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:i*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿig
"dense_712/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_712/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_712/kernel/Regularizer/AbsAbs7dense_712/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_712/kernel/Regularizer/SumSum$dense_712/kernel/Regularizer/Abs:y:0-dense_712/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_712/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_712/kernel/Regularizer/mulMul+dense_712/kernel/Regularizer/mul/x:output:0)dense_712/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_712/kernel/Regularizer/addAddV2+dense_712/kernel/Regularizer/Const:output:0$dense_712/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_712/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_712/kernel/Regularizer/SquareSquare:dense_712/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_712/kernel/Regularizer/Sum_1Sum'dense_712/kernel/Regularizer/Square:y:0-dense_712/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_712/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_712/kernel/Regularizer/mul_1Mul-dense_712/kernel/Regularizer/mul_1/x:output:0+dense_712/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_712/kernel/Regularizer/add_1AddV2$dense_712/kernel/Regularizer/add:z:0&dense_712/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÞ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_712/kernel/Regularizer/Abs/ReadVariableOp3^dense_712/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_712/kernel/Regularizer/Abs/ReadVariableOp/dense_712/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_712/kernel/Regularizer/Square/ReadVariableOp2dense_712/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_647_layer_call_fn_1126025

inputs
unknown:7
	unknown_0:7
	unknown_1:7
	unknown_2:7
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_647_layer_call_and_return_conditional_losses_1121406o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
¥
Þ
F__inference_dense_716_layer_call_and_return_conditional_losses_1121773

inputs0
matmul_readvariableop_resource:i=-
biasadd_readvariableop_resource:=
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_716/kernel/Regularizer/Abs/ReadVariableOp¢2dense_716/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:i=*
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
:ÿÿÿÿÿÿÿÿÿ=g
"dense_716/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_716/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:i=*
dtype0
 dense_716/kernel/Regularizer/AbsAbs7dense_716/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_716/kernel/Regularizer/SumSum$dense_716/kernel/Regularizer/Abs:y:0-dense_716/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_716/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_716/kernel/Regularizer/mulMul+dense_716/kernel/Regularizer/mul/x:output:0)dense_716/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_716/kernel/Regularizer/addAddV2+dense_716/kernel/Regularizer/Const:output:0$dense_716/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_716/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:i=*
dtype0
#dense_716/kernel/Regularizer/SquareSquare:dense_716/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_716/kernel/Regularizer/Sum_1Sum'dense_716/kernel/Regularizer/Square:y:0-dense_716/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_716/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_716/kernel/Regularizer/mul_1Mul-dense_716/kernel/Regularizer/mul_1/x:output:0+dense_716/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_716/kernel/Regularizer/add_1AddV2$dense_716/kernel/Regularizer/add:z:0&dense_716/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=Þ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_716/kernel/Regularizer/Abs/ReadVariableOp3^dense_716/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_716/kernel/Regularizer/Abs/ReadVariableOp/dense_716/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_716/kernel/Regularizer/Square/ReadVariableOp2dense_716/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_643_layer_call_and_return_conditional_losses_1121699

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿi:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_647_layer_call_and_return_conditional_losses_1126079

inputs5
'assignmovingavg_readvariableop_resource:77
)assignmovingavg_1_readvariableop_resource:73
%batchnorm_mul_readvariableop_resource:7/
!batchnorm_readvariableop_resource:7
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:7
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:7*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:7*
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
:7*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:7x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:7¬
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
:7*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:7~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:7´
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
:7P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:7~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:7v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:7r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs

ã
__inference_loss_fn_8_1126427J
8dense_719_kernel_regularizer_abs_readvariableop_resource:77
identity¢/dense_719/kernel/Regularizer/Abs/ReadVariableOp¢2dense_719/kernel/Regularizer/Square/ReadVariableOpg
"dense_719/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
/dense_719/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8dense_719_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:77*
dtype0
 dense_719/kernel/Regularizer/AbsAbs7dense_719/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_719/kernel/Regularizer/SumSum$dense_719/kernel/Regularizer/Abs:y:0-dense_719/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_719/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_719/kernel/Regularizer/mulMul+dense_719/kernel/Regularizer/mul/x:output:0)dense_719/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_719/kernel/Regularizer/addAddV2+dense_719/kernel/Regularizer/Const:output:0$dense_719/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: «
2dense_719/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense_719_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:77*
dtype0
#dense_719/kernel/Regularizer/SquareSquare:dense_719/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_719/kernel/Regularizer/Sum_1Sum'dense_719/kernel/Regularizer/Square:y:0-dense_719/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_719/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_719/kernel/Regularizer/mul_1Mul-dense_719/kernel/Regularizer/mul_1/x:output:0+dense_719/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_719/kernel/Regularizer/add_1AddV2$dense_719/kernel/Regularizer/add:z:0&dense_719/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
IdentityIdentity&dense_719/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ­
NoOpNoOp0^dense_719/kernel/Regularizer/Abs/ReadVariableOp3^dense_719/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_719/kernel/Regularizer/Abs/ReadVariableOp/dense_719/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_719/kernel/Regularizer/Square/ReadVariableOp2dense_719/kernel/Regularizer/Square/ReadVariableOp
­
M
1__inference_leaky_re_lu_648_layer_call_fn_1126223

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
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_648_layer_call_and_return_conditional_losses_1121934`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_640_layer_call_fn_1125052

inputs
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_640_layer_call_and_return_conditional_losses_1120832o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_648_layer_call_and_return_conditional_losses_1121934

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
éÆ
Ã!
J__inference_sequential_71_layer_call_and_return_conditional_losses_1122770

inputs
normalization_71_sub_y
normalization_71_sqrt_x#
dense_711_1122494:i
dense_711_1122496:i-
batch_normalization_640_1122499:i-
batch_normalization_640_1122501:i-
batch_normalization_640_1122503:i-
batch_normalization_640_1122505:i#
dense_712_1122509:ii
dense_712_1122511:i-
batch_normalization_641_1122514:i-
batch_normalization_641_1122516:i-
batch_normalization_641_1122518:i-
batch_normalization_641_1122520:i#
dense_713_1122524:ii
dense_713_1122526:i-
batch_normalization_642_1122529:i-
batch_normalization_642_1122531:i-
batch_normalization_642_1122533:i-
batch_normalization_642_1122535:i#
dense_714_1122539:ii
dense_714_1122541:i-
batch_normalization_643_1122544:i-
batch_normalization_643_1122546:i-
batch_normalization_643_1122548:i-
batch_normalization_643_1122550:i#
dense_715_1122554:ii
dense_715_1122556:i-
batch_normalization_644_1122559:i-
batch_normalization_644_1122561:i-
batch_normalization_644_1122563:i-
batch_normalization_644_1122565:i#
dense_716_1122569:i=
dense_716_1122571:=-
batch_normalization_645_1122574:=-
batch_normalization_645_1122576:=-
batch_normalization_645_1122578:=-
batch_normalization_645_1122580:=#
dense_717_1122584:==
dense_717_1122586:=-
batch_normalization_646_1122589:=-
batch_normalization_646_1122591:=-
batch_normalization_646_1122593:=-
batch_normalization_646_1122595:=#
dense_718_1122599:=7
dense_718_1122601:7-
batch_normalization_647_1122604:7-
batch_normalization_647_1122606:7-
batch_normalization_647_1122608:7-
batch_normalization_647_1122610:7#
dense_719_1122614:77
dense_719_1122616:7-
batch_normalization_648_1122619:7-
batch_normalization_648_1122621:7-
batch_normalization_648_1122623:7-
batch_normalization_648_1122625:7#
dense_720_1122629:7
dense_720_1122631:
identity¢/batch_normalization_640/StatefulPartitionedCall¢/batch_normalization_641/StatefulPartitionedCall¢/batch_normalization_642/StatefulPartitionedCall¢/batch_normalization_643/StatefulPartitionedCall¢/batch_normalization_644/StatefulPartitionedCall¢/batch_normalization_645/StatefulPartitionedCall¢/batch_normalization_646/StatefulPartitionedCall¢/batch_normalization_647/StatefulPartitionedCall¢/batch_normalization_648/StatefulPartitionedCall¢!dense_711/StatefulPartitionedCall¢/dense_711/kernel/Regularizer/Abs/ReadVariableOp¢2dense_711/kernel/Regularizer/Square/ReadVariableOp¢!dense_712/StatefulPartitionedCall¢/dense_712/kernel/Regularizer/Abs/ReadVariableOp¢2dense_712/kernel/Regularizer/Square/ReadVariableOp¢!dense_713/StatefulPartitionedCall¢/dense_713/kernel/Regularizer/Abs/ReadVariableOp¢2dense_713/kernel/Regularizer/Square/ReadVariableOp¢!dense_714/StatefulPartitionedCall¢/dense_714/kernel/Regularizer/Abs/ReadVariableOp¢2dense_714/kernel/Regularizer/Square/ReadVariableOp¢!dense_715/StatefulPartitionedCall¢/dense_715/kernel/Regularizer/Abs/ReadVariableOp¢2dense_715/kernel/Regularizer/Square/ReadVariableOp¢!dense_716/StatefulPartitionedCall¢/dense_716/kernel/Regularizer/Abs/ReadVariableOp¢2dense_716/kernel/Regularizer/Square/ReadVariableOp¢!dense_717/StatefulPartitionedCall¢/dense_717/kernel/Regularizer/Abs/ReadVariableOp¢2dense_717/kernel/Regularizer/Square/ReadVariableOp¢!dense_718/StatefulPartitionedCall¢/dense_718/kernel/Regularizer/Abs/ReadVariableOp¢2dense_718/kernel/Regularizer/Square/ReadVariableOp¢!dense_719/StatefulPartitionedCall¢/dense_719/kernel/Regularizer/Abs/ReadVariableOp¢2dense_719/kernel/Regularizer/Square/ReadVariableOp¢!dense_720/StatefulPartitionedCallm
normalization_71/subSubinputsnormalization_71_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_71/SqrtSqrtnormalization_71_sqrt_x*
T0*
_output_shapes

:_
normalization_71/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_71/MaximumMaximumnormalization_71/Sqrt:y:0#normalization_71/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_71/truedivRealDivnormalization_71/sub:z:0normalization_71/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_711/StatefulPartitionedCallStatefulPartitionedCallnormalization_71/truediv:z:0dense_711_1122494dense_711_1122496*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_711_layer_call_and_return_conditional_losses_1121538
/batch_normalization_640/StatefulPartitionedCallStatefulPartitionedCall*dense_711/StatefulPartitionedCall:output:0batch_normalization_640_1122499batch_normalization_640_1122501batch_normalization_640_1122503batch_normalization_640_1122505*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_640_layer_call_and_return_conditional_losses_1120832ù
leaky_re_lu_640/PartitionedCallPartitionedCall8batch_normalization_640/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_640_layer_call_and_return_conditional_losses_1121558
!dense_712/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_640/PartitionedCall:output:0dense_712_1122509dense_712_1122511*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_712_layer_call_and_return_conditional_losses_1121585
/batch_normalization_641/StatefulPartitionedCallStatefulPartitionedCall*dense_712/StatefulPartitionedCall:output:0batch_normalization_641_1122514batch_normalization_641_1122516batch_normalization_641_1122518batch_normalization_641_1122520*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_641_layer_call_and_return_conditional_losses_1120914ù
leaky_re_lu_641/PartitionedCallPartitionedCall8batch_normalization_641/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_641_layer_call_and_return_conditional_losses_1121605
!dense_713/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_641/PartitionedCall:output:0dense_713_1122524dense_713_1122526*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_713_layer_call_and_return_conditional_losses_1121632
/batch_normalization_642/StatefulPartitionedCallStatefulPartitionedCall*dense_713/StatefulPartitionedCall:output:0batch_normalization_642_1122529batch_normalization_642_1122531batch_normalization_642_1122533batch_normalization_642_1122535*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_642_layer_call_and_return_conditional_losses_1120996ù
leaky_re_lu_642/PartitionedCallPartitionedCall8batch_normalization_642/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_642_layer_call_and_return_conditional_losses_1121652
!dense_714/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_642/PartitionedCall:output:0dense_714_1122539dense_714_1122541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_714_layer_call_and_return_conditional_losses_1121679
/batch_normalization_643/StatefulPartitionedCallStatefulPartitionedCall*dense_714/StatefulPartitionedCall:output:0batch_normalization_643_1122544batch_normalization_643_1122546batch_normalization_643_1122548batch_normalization_643_1122550*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_643_layer_call_and_return_conditional_losses_1121078ù
leaky_re_lu_643/PartitionedCallPartitionedCall8batch_normalization_643/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_643_layer_call_and_return_conditional_losses_1121699
!dense_715/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_643/PartitionedCall:output:0dense_715_1122554dense_715_1122556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_715_layer_call_and_return_conditional_losses_1121726
/batch_normalization_644/StatefulPartitionedCallStatefulPartitionedCall*dense_715/StatefulPartitionedCall:output:0batch_normalization_644_1122559batch_normalization_644_1122561batch_normalization_644_1122563batch_normalization_644_1122565*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_644_layer_call_and_return_conditional_losses_1121160ù
leaky_re_lu_644/PartitionedCallPartitionedCall8batch_normalization_644/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_644_layer_call_and_return_conditional_losses_1121746
!dense_716/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_644/PartitionedCall:output:0dense_716_1122569dense_716_1122571*
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
GPU 2J 8 *O
fJRH
F__inference_dense_716_layer_call_and_return_conditional_losses_1121773
/batch_normalization_645/StatefulPartitionedCallStatefulPartitionedCall*dense_716/StatefulPartitionedCall:output:0batch_normalization_645_1122574batch_normalization_645_1122576batch_normalization_645_1122578batch_normalization_645_1122580*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_645_layer_call_and_return_conditional_losses_1121242ù
leaky_re_lu_645/PartitionedCallPartitionedCall8batch_normalization_645/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_645_layer_call_and_return_conditional_losses_1121793
!dense_717/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_645/PartitionedCall:output:0dense_717_1122584dense_717_1122586*
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
GPU 2J 8 *O
fJRH
F__inference_dense_717_layer_call_and_return_conditional_losses_1121820
/batch_normalization_646/StatefulPartitionedCallStatefulPartitionedCall*dense_717/StatefulPartitionedCall:output:0batch_normalization_646_1122589batch_normalization_646_1122591batch_normalization_646_1122593batch_normalization_646_1122595*
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_646_layer_call_and_return_conditional_losses_1121324ù
leaky_re_lu_646/PartitionedCallPartitionedCall8batch_normalization_646/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_646_layer_call_and_return_conditional_losses_1121840
!dense_718/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_646/PartitionedCall:output:0dense_718_1122599dense_718_1122601*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_718_layer_call_and_return_conditional_losses_1121867
/batch_normalization_647/StatefulPartitionedCallStatefulPartitionedCall*dense_718/StatefulPartitionedCall:output:0batch_normalization_647_1122604batch_normalization_647_1122606batch_normalization_647_1122608batch_normalization_647_1122610*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_647_layer_call_and_return_conditional_losses_1121406ù
leaky_re_lu_647/PartitionedCallPartitionedCall8batch_normalization_647/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_647_layer_call_and_return_conditional_losses_1121887
!dense_719/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_647/PartitionedCall:output:0dense_719_1122614dense_719_1122616*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_719_layer_call_and_return_conditional_losses_1121914
/batch_normalization_648/StatefulPartitionedCallStatefulPartitionedCall*dense_719/StatefulPartitionedCall:output:0batch_normalization_648_1122619batch_normalization_648_1122621batch_normalization_648_1122623batch_normalization_648_1122625*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_648_layer_call_and_return_conditional_losses_1121488ù
leaky_re_lu_648/PartitionedCallPartitionedCall8batch_normalization_648/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_648_layer_call_and_return_conditional_losses_1121934
!dense_720/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_648/PartitionedCall:output:0dense_720_1122629dense_720_1122631*
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
F__inference_dense_720_layer_call_and_return_conditional_losses_1121946g
"dense_711/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_711/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_711_1122494*
_output_shapes

:i*
dtype0
 dense_711/kernel/Regularizer/AbsAbs7dense_711/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_711/kernel/Regularizer/SumSum$dense_711/kernel/Regularizer/Abs:y:0-dense_711/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_711/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_711/kernel/Regularizer/mulMul+dense_711/kernel/Regularizer/mul/x:output:0)dense_711/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_711/kernel/Regularizer/addAddV2+dense_711/kernel/Regularizer/Const:output:0$dense_711/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_711/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_711_1122494*
_output_shapes

:i*
dtype0
#dense_711/kernel/Regularizer/SquareSquare:dense_711/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_711/kernel/Regularizer/Sum_1Sum'dense_711/kernel/Regularizer/Square:y:0-dense_711/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_711/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_711/kernel/Regularizer/mul_1Mul-dense_711/kernel/Regularizer/mul_1/x:output:0+dense_711/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_711/kernel/Regularizer/add_1AddV2$dense_711/kernel/Regularizer/add:z:0&dense_711/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_712/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_712/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_712_1122509*
_output_shapes

:ii*
dtype0
 dense_712/kernel/Regularizer/AbsAbs7dense_712/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_712/kernel/Regularizer/SumSum$dense_712/kernel/Regularizer/Abs:y:0-dense_712/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_712/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_712/kernel/Regularizer/mulMul+dense_712/kernel/Regularizer/mul/x:output:0)dense_712/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_712/kernel/Regularizer/addAddV2+dense_712/kernel/Regularizer/Const:output:0$dense_712/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_712/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_712_1122509*
_output_shapes

:ii*
dtype0
#dense_712/kernel/Regularizer/SquareSquare:dense_712/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_712/kernel/Regularizer/Sum_1Sum'dense_712/kernel/Regularizer/Square:y:0-dense_712/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_712/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_712/kernel/Regularizer/mul_1Mul-dense_712/kernel/Regularizer/mul_1/x:output:0+dense_712/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_712/kernel/Regularizer/add_1AddV2$dense_712/kernel/Regularizer/add:z:0&dense_712/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_713/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_713/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_713_1122524*
_output_shapes

:ii*
dtype0
 dense_713/kernel/Regularizer/AbsAbs7dense_713/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_713/kernel/Regularizer/SumSum$dense_713/kernel/Regularizer/Abs:y:0-dense_713/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_713/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_713/kernel/Regularizer/mulMul+dense_713/kernel/Regularizer/mul/x:output:0)dense_713/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_713/kernel/Regularizer/addAddV2+dense_713/kernel/Regularizer/Const:output:0$dense_713/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_713/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_713_1122524*
_output_shapes

:ii*
dtype0
#dense_713/kernel/Regularizer/SquareSquare:dense_713/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_713/kernel/Regularizer/Sum_1Sum'dense_713/kernel/Regularizer/Square:y:0-dense_713/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_713/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_713/kernel/Regularizer/mul_1Mul-dense_713/kernel/Regularizer/mul_1/x:output:0+dense_713/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_713/kernel/Regularizer/add_1AddV2$dense_713/kernel/Regularizer/add:z:0&dense_713/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_714/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_714/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_714_1122539*
_output_shapes

:ii*
dtype0
 dense_714/kernel/Regularizer/AbsAbs7dense_714/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_714/kernel/Regularizer/SumSum$dense_714/kernel/Regularizer/Abs:y:0-dense_714/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_714/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_714/kernel/Regularizer/mulMul+dense_714/kernel/Regularizer/mul/x:output:0)dense_714/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_714/kernel/Regularizer/addAddV2+dense_714/kernel/Regularizer/Const:output:0$dense_714/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_714/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_714_1122539*
_output_shapes

:ii*
dtype0
#dense_714/kernel/Regularizer/SquareSquare:dense_714/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_714/kernel/Regularizer/Sum_1Sum'dense_714/kernel/Regularizer/Square:y:0-dense_714/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_714/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_714/kernel/Regularizer/mul_1Mul-dense_714/kernel/Regularizer/mul_1/x:output:0+dense_714/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_714/kernel/Regularizer/add_1AddV2$dense_714/kernel/Regularizer/add:z:0&dense_714/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_715/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_715/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_715_1122554*
_output_shapes

:ii*
dtype0
 dense_715/kernel/Regularizer/AbsAbs7dense_715/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_715/kernel/Regularizer/SumSum$dense_715/kernel/Regularizer/Abs:y:0-dense_715/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_715/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_715/kernel/Regularizer/mulMul+dense_715/kernel/Regularizer/mul/x:output:0)dense_715/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_715/kernel/Regularizer/addAddV2+dense_715/kernel/Regularizer/Const:output:0$dense_715/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_715/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_715_1122554*
_output_shapes

:ii*
dtype0
#dense_715/kernel/Regularizer/SquareSquare:dense_715/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_715/kernel/Regularizer/Sum_1Sum'dense_715/kernel/Regularizer/Square:y:0-dense_715/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_715/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_715/kernel/Regularizer/mul_1Mul-dense_715/kernel/Regularizer/mul_1/x:output:0+dense_715/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_715/kernel/Regularizer/add_1AddV2$dense_715/kernel/Regularizer/add:z:0&dense_715/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_716/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_716/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_716_1122569*
_output_shapes

:i=*
dtype0
 dense_716/kernel/Regularizer/AbsAbs7dense_716/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_716/kernel/Regularizer/SumSum$dense_716/kernel/Regularizer/Abs:y:0-dense_716/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_716/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_716/kernel/Regularizer/mulMul+dense_716/kernel/Regularizer/mul/x:output:0)dense_716/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_716/kernel/Regularizer/addAddV2+dense_716/kernel/Regularizer/Const:output:0$dense_716/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_716/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_716_1122569*
_output_shapes

:i=*
dtype0
#dense_716/kernel/Regularizer/SquareSquare:dense_716/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_716/kernel/Regularizer/Sum_1Sum'dense_716/kernel/Regularizer/Square:y:0-dense_716/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_716/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_716/kernel/Regularizer/mul_1Mul-dense_716/kernel/Regularizer/mul_1/x:output:0+dense_716/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_716/kernel/Regularizer/add_1AddV2$dense_716/kernel/Regularizer/add:z:0&dense_716/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_717/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_717/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_717_1122584*
_output_shapes

:==*
dtype0
 dense_717/kernel/Regularizer/AbsAbs7dense_717/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_717/kernel/Regularizer/SumSum$dense_717/kernel/Regularizer/Abs:y:0-dense_717/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_717/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_717/kernel/Regularizer/mulMul+dense_717/kernel/Regularizer/mul/x:output:0)dense_717/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_717/kernel/Regularizer/addAddV2+dense_717/kernel/Regularizer/Const:output:0$dense_717/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_717/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_717_1122584*
_output_shapes

:==*
dtype0
#dense_717/kernel/Regularizer/SquareSquare:dense_717/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_717/kernel/Regularizer/Sum_1Sum'dense_717/kernel/Regularizer/Square:y:0-dense_717/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_717/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_717/kernel/Regularizer/mul_1Mul-dense_717/kernel/Regularizer/mul_1/x:output:0+dense_717/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_717/kernel/Regularizer/add_1AddV2$dense_717/kernel/Regularizer/add:z:0&dense_717/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_718/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_718/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_718_1122599*
_output_shapes

:=7*
dtype0
 dense_718/kernel/Regularizer/AbsAbs7dense_718/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_718/kernel/Regularizer/SumSum$dense_718/kernel/Regularizer/Abs:y:0-dense_718/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_718/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_718/kernel/Regularizer/mulMul+dense_718/kernel/Regularizer/mul/x:output:0)dense_718/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_718/kernel/Regularizer/addAddV2+dense_718/kernel/Regularizer/Const:output:0$dense_718/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_718/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_718_1122599*
_output_shapes

:=7*
dtype0
#dense_718/kernel/Regularizer/SquareSquare:dense_718/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_718/kernel/Regularizer/Sum_1Sum'dense_718/kernel/Regularizer/Square:y:0-dense_718/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_718/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_718/kernel/Regularizer/mul_1Mul-dense_718/kernel/Regularizer/mul_1/x:output:0+dense_718/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_718/kernel/Regularizer/add_1AddV2$dense_718/kernel/Regularizer/add:z:0&dense_718/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_719/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_719/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_719_1122614*
_output_shapes

:77*
dtype0
 dense_719/kernel/Regularizer/AbsAbs7dense_719/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_719/kernel/Regularizer/SumSum$dense_719/kernel/Regularizer/Abs:y:0-dense_719/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_719/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_719/kernel/Regularizer/mulMul+dense_719/kernel/Regularizer/mul/x:output:0)dense_719/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_719/kernel/Regularizer/addAddV2+dense_719/kernel/Regularizer/Const:output:0$dense_719/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_719/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_719_1122614*
_output_shapes

:77*
dtype0
#dense_719/kernel/Regularizer/SquareSquare:dense_719/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_719/kernel/Regularizer/Sum_1Sum'dense_719/kernel/Regularizer/Square:y:0-dense_719/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_719/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_719/kernel/Regularizer/mul_1Mul-dense_719/kernel/Regularizer/mul_1/x:output:0+dense_719/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_719/kernel/Regularizer/add_1AddV2$dense_719/kernel/Regularizer/add:z:0&dense_719/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_720/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_640/StatefulPartitionedCall0^batch_normalization_641/StatefulPartitionedCall0^batch_normalization_642/StatefulPartitionedCall0^batch_normalization_643/StatefulPartitionedCall0^batch_normalization_644/StatefulPartitionedCall0^batch_normalization_645/StatefulPartitionedCall0^batch_normalization_646/StatefulPartitionedCall0^batch_normalization_647/StatefulPartitionedCall0^batch_normalization_648/StatefulPartitionedCall"^dense_711/StatefulPartitionedCall0^dense_711/kernel/Regularizer/Abs/ReadVariableOp3^dense_711/kernel/Regularizer/Square/ReadVariableOp"^dense_712/StatefulPartitionedCall0^dense_712/kernel/Regularizer/Abs/ReadVariableOp3^dense_712/kernel/Regularizer/Square/ReadVariableOp"^dense_713/StatefulPartitionedCall0^dense_713/kernel/Regularizer/Abs/ReadVariableOp3^dense_713/kernel/Regularizer/Square/ReadVariableOp"^dense_714/StatefulPartitionedCall0^dense_714/kernel/Regularizer/Abs/ReadVariableOp3^dense_714/kernel/Regularizer/Square/ReadVariableOp"^dense_715/StatefulPartitionedCall0^dense_715/kernel/Regularizer/Abs/ReadVariableOp3^dense_715/kernel/Regularizer/Square/ReadVariableOp"^dense_716/StatefulPartitionedCall0^dense_716/kernel/Regularizer/Abs/ReadVariableOp3^dense_716/kernel/Regularizer/Square/ReadVariableOp"^dense_717/StatefulPartitionedCall0^dense_717/kernel/Regularizer/Abs/ReadVariableOp3^dense_717/kernel/Regularizer/Square/ReadVariableOp"^dense_718/StatefulPartitionedCall0^dense_718/kernel/Regularizer/Abs/ReadVariableOp3^dense_718/kernel/Regularizer/Square/ReadVariableOp"^dense_719/StatefulPartitionedCall0^dense_719/kernel/Regularizer/Abs/ReadVariableOp3^dense_719/kernel/Regularizer/Square/ReadVariableOp"^dense_720/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_640/StatefulPartitionedCall/batch_normalization_640/StatefulPartitionedCall2b
/batch_normalization_641/StatefulPartitionedCall/batch_normalization_641/StatefulPartitionedCall2b
/batch_normalization_642/StatefulPartitionedCall/batch_normalization_642/StatefulPartitionedCall2b
/batch_normalization_643/StatefulPartitionedCall/batch_normalization_643/StatefulPartitionedCall2b
/batch_normalization_644/StatefulPartitionedCall/batch_normalization_644/StatefulPartitionedCall2b
/batch_normalization_645/StatefulPartitionedCall/batch_normalization_645/StatefulPartitionedCall2b
/batch_normalization_646/StatefulPartitionedCall/batch_normalization_646/StatefulPartitionedCall2b
/batch_normalization_647/StatefulPartitionedCall/batch_normalization_647/StatefulPartitionedCall2b
/batch_normalization_648/StatefulPartitionedCall/batch_normalization_648/StatefulPartitionedCall2F
!dense_711/StatefulPartitionedCall!dense_711/StatefulPartitionedCall2b
/dense_711/kernel/Regularizer/Abs/ReadVariableOp/dense_711/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_711/kernel/Regularizer/Square/ReadVariableOp2dense_711/kernel/Regularizer/Square/ReadVariableOp2F
!dense_712/StatefulPartitionedCall!dense_712/StatefulPartitionedCall2b
/dense_712/kernel/Regularizer/Abs/ReadVariableOp/dense_712/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_712/kernel/Regularizer/Square/ReadVariableOp2dense_712/kernel/Regularizer/Square/ReadVariableOp2F
!dense_713/StatefulPartitionedCall!dense_713/StatefulPartitionedCall2b
/dense_713/kernel/Regularizer/Abs/ReadVariableOp/dense_713/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_713/kernel/Regularizer/Square/ReadVariableOp2dense_713/kernel/Regularizer/Square/ReadVariableOp2F
!dense_714/StatefulPartitionedCall!dense_714/StatefulPartitionedCall2b
/dense_714/kernel/Regularizer/Abs/ReadVariableOp/dense_714/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_714/kernel/Regularizer/Square/ReadVariableOp2dense_714/kernel/Regularizer/Square/ReadVariableOp2F
!dense_715/StatefulPartitionedCall!dense_715/StatefulPartitionedCall2b
/dense_715/kernel/Regularizer/Abs/ReadVariableOp/dense_715/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_715/kernel/Regularizer/Square/ReadVariableOp2dense_715/kernel/Regularizer/Square/ReadVariableOp2F
!dense_716/StatefulPartitionedCall!dense_716/StatefulPartitionedCall2b
/dense_716/kernel/Regularizer/Abs/ReadVariableOp/dense_716/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_716/kernel/Regularizer/Square/ReadVariableOp2dense_716/kernel/Regularizer/Square/ReadVariableOp2F
!dense_717/StatefulPartitionedCall!dense_717/StatefulPartitionedCall2b
/dense_717/kernel/Regularizer/Abs/ReadVariableOp/dense_717/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_717/kernel/Regularizer/Square/ReadVariableOp2dense_717/kernel/Regularizer/Square/ReadVariableOp2F
!dense_718/StatefulPartitionedCall!dense_718/StatefulPartitionedCall2b
/dense_718/kernel/Regularizer/Abs/ReadVariableOp/dense_718/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_718/kernel/Regularizer/Square/ReadVariableOp2dense_718/kernel/Regularizer/Square/ReadVariableOp2F
!dense_719/StatefulPartitionedCall!dense_719/StatefulPartitionedCall2b
/dense_719/kernel/Regularizer/Abs/ReadVariableOp/dense_719/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_719/kernel/Regularizer/Square/ReadVariableOp2dense_719/kernel/Regularizer/Square/ReadVariableOp2F
!dense_720/StatefulPartitionedCall!dense_720/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
­
M
1__inference_leaky_re_lu_647_layer_call_fn_1126084

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
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_647_layer_call_and_return_conditional_losses_1121887`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_642_layer_call_and_return_conditional_losses_1120996

inputs5
'assignmovingavg_readvariableop_resource:i7
)assignmovingavg_1_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i/
!batchnorm_readvariableop_resource:i
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:i
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿil
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:i*
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
:i*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ix
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i¬
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
:i*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:i~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i´
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿih
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:iv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_646_layer_call_and_return_conditional_losses_1121277

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
Æ

+__inference_dense_715_layer_call_fn_1125557

inputs
unknown:ii
	unknown_0:i
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_715_layer_call_and_return_conditional_losses_1121726o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_648_layer_call_fn_1126151

inputs
unknown:7
	unknown_0:7
	unknown_1:7
	unknown_2:7
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_648_layer_call_and_return_conditional_losses_1121441o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
¥
Þ
F__inference_dense_714_layer_call_and_return_conditional_losses_1121679

inputs0
matmul_readvariableop_resource:ii-
biasadd_readvariableop_resource:i
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_714/kernel/Regularizer/Abs/ReadVariableOp¢2dense_714/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿir
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:i*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿig
"dense_714/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_714/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_714/kernel/Regularizer/AbsAbs7dense_714/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_714/kernel/Regularizer/SumSum$dense_714/kernel/Regularizer/Abs:y:0-dense_714/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_714/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_714/kernel/Regularizer/mulMul+dense_714/kernel/Regularizer/mul/x:output:0)dense_714/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_714/kernel/Regularizer/addAddV2+dense_714/kernel/Regularizer/Const:output:0$dense_714/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_714/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_714/kernel/Regularizer/SquareSquare:dense_714/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_714/kernel/Regularizer/Sum_1Sum'dense_714/kernel/Regularizer/Square:y:0-dense_714/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_714/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_714/kernel/Regularizer/mul_1Mul-dense_714/kernel/Regularizer/mul_1/x:output:0+dense_714/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_714/kernel/Regularizer/add_1AddV2$dense_714/kernel/Regularizer/add:z:0&dense_714/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÞ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_714/kernel/Regularizer/Abs/ReadVariableOp3^dense_714/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_714/kernel/Regularizer/Abs/ReadVariableOp/dense_714/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_714/kernel/Regularizer/Square/ReadVariableOp2dense_714/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_643_layer_call_fn_1125456

inputs
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_643_layer_call_and_return_conditional_losses_1121031o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
Æ

+__inference_dense_719_layer_call_fn_1126113

inputs
unknown:77
	unknown_0:7
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_719_layer_call_and_return_conditional_losses_1121914o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs

ã
__inference_loss_fn_7_1126407J
8dense_718_kernel_regularizer_abs_readvariableop_resource:=7
identity¢/dense_718/kernel/Regularizer/Abs/ReadVariableOp¢2dense_718/kernel/Regularizer/Square/ReadVariableOpg
"dense_718/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
/dense_718/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8dense_718_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:=7*
dtype0
 dense_718/kernel/Regularizer/AbsAbs7dense_718/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_718/kernel/Regularizer/SumSum$dense_718/kernel/Regularizer/Abs:y:0-dense_718/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_718/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_718/kernel/Regularizer/mulMul+dense_718/kernel/Regularizer/mul/x:output:0)dense_718/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_718/kernel/Regularizer/addAddV2+dense_718/kernel/Regularizer/Const:output:0$dense_718/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: «
2dense_718/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense_718_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:=7*
dtype0
#dense_718/kernel/Regularizer/SquareSquare:dense_718/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_718/kernel/Regularizer/Sum_1Sum'dense_718/kernel/Regularizer/Square:y:0-dense_718/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_718/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_718/kernel/Regularizer/mul_1Mul-dense_718/kernel/Regularizer/mul_1/x:output:0+dense_718/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_718/kernel/Regularizer/add_1AddV2$dense_718/kernel/Regularizer/add:z:0&dense_718/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
IdentityIdentity&dense_718/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ­
NoOpNoOp0^dense_718/kernel/Regularizer/Abs/ReadVariableOp3^dense_718/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_718/kernel/Regularizer/Abs/ReadVariableOp/dense_718/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_718/kernel/Regularizer/Square/ReadVariableOp2dense_718/kernel/Regularizer/Square/ReadVariableOp
Ñ
³
T__inference_batch_normalization_641_layer_call_and_return_conditional_losses_1125211

inputs/
!batchnorm_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i1
#batchnorm_readvariableop_1_resource:i1
#batchnorm_readvariableop_2_resource:i
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:iz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs

ã
__inference_loss_fn_3_1126327J
8dense_714_kernel_regularizer_abs_readvariableop_resource:ii
identity¢/dense_714/kernel/Regularizer/Abs/ReadVariableOp¢2dense_714/kernel/Regularizer/Square/ReadVariableOpg
"dense_714/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
/dense_714/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8dense_714_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_714/kernel/Regularizer/AbsAbs7dense_714/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_714/kernel/Regularizer/SumSum$dense_714/kernel/Regularizer/Abs:y:0-dense_714/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_714/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_714/kernel/Regularizer/mulMul+dense_714/kernel/Regularizer/mul/x:output:0)dense_714/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_714/kernel/Regularizer/addAddV2+dense_714/kernel/Regularizer/Const:output:0$dense_714/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: «
2dense_714/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense_714_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_714/kernel/Regularizer/SquareSquare:dense_714/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_714/kernel/Regularizer/Sum_1Sum'dense_714/kernel/Regularizer/Square:y:0-dense_714/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_714/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_714/kernel/Regularizer/mul_1Mul-dense_714/kernel/Regularizer/mul_1/x:output:0+dense_714/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_714/kernel/Regularizer/add_1AddV2$dense_714/kernel/Regularizer/add:z:0&dense_714/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
IdentityIdentity&dense_714/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ­
NoOpNoOp0^dense_714/kernel/Regularizer/Abs/ReadVariableOp3^dense_714/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_714/kernel/Regularizer/Abs/ReadVariableOp/dense_714/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_714/kernel/Regularizer/Square/ReadVariableOp2dense_714/kernel/Regularizer/Square/ReadVariableOp
%
í
T__inference_batch_normalization_641_layer_call_and_return_conditional_losses_1120914

inputs5
'assignmovingavg_readvariableop_resource:i7
)assignmovingavg_1_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i/
!batchnorm_readvariableop_resource:i
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:i
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿil
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:i*
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
:i*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ix
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i¬
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
:i*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:i~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i´
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿih
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:iv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
É	
÷
F__inference_dense_720_layer_call_and_return_conditional_losses_1126247

inputs0
matmul_readvariableop_resource:7-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7*
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
:ÿÿÿÿÿÿÿÿÿ7: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_644_layer_call_and_return_conditional_losses_1125672

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿi:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
Æ

+__inference_dense_713_layer_call_fn_1125279

inputs
unknown:ii
	unknown_0:i
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_713_layer_call_and_return_conditional_losses_1121632o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_648_layer_call_fn_1126164

inputs
unknown:7
	unknown_0:7
	unknown_1:7
	unknown_2:7
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_648_layer_call_and_return_conditional_losses_1121488o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
Æ

+__inference_dense_716_layer_call_fn_1125696

inputs
unknown:i=
	unknown_0:=
identity¢StatefulPartitionedCallÛ
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
GPU 2J 8 *O
fJRH
F__inference_dense_716_layer_call_and_return_conditional_losses_1121773o
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
:ÿÿÿÿÿÿÿÿÿi: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_646_layer_call_fn_1125873

inputs
unknown:=
	unknown_0:=
	unknown_1:=
	unknown_2:=
identity¢StatefulPartitionedCall
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_646_layer_call_and_return_conditional_losses_1121277o
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
Æ

+__inference_dense_717_layer_call_fn_1125835

inputs
unknown:==
	unknown_0:=
identity¢StatefulPartitionedCallÛ
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
GPU 2J 8 *O
fJRH
F__inference_dense_717_layer_call_and_return_conditional_losses_1121820o
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
É	
÷
F__inference_dense_720_layer_call_and_return_conditional_losses_1121946

inputs0
matmul_readvariableop_resource:7-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7*
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
:ÿÿÿÿÿÿÿÿÿ7: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_640_layer_call_and_return_conditional_losses_1121558

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿi:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_647_layer_call_and_return_conditional_losses_1121359

inputs/
!batchnorm_readvariableop_resource:73
%batchnorm_mul_readvariableop_resource:71
#batchnorm_readvariableop_1_resource:71
#batchnorm_readvariableop_2_resource:7
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:7*
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
:7P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:7~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:7*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:7z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:7*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:7r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_647_layer_call_and_return_conditional_losses_1121406

inputs5
'assignmovingavg_readvariableop_resource:77
)assignmovingavg_1_readvariableop_resource:73
%batchnorm_mul_readvariableop_resource:7/
!batchnorm_readvariableop_resource:7
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:7
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:7*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:7*
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
:7*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:7x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:7¬
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
:7*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:7~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:7´
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
:7P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:7~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:7v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:7r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_644_layer_call_fn_1125608

inputs
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_644_layer_call_and_return_conditional_losses_1121160o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
Æ

+__inference_dense_714_layer_call_fn_1125418

inputs
unknown:ii
	unknown_0:i
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_714_layer_call_and_return_conditional_losses_1121679o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
¥
Þ
F__inference_dense_717_layer_call_and_return_conditional_losses_1121820

inputs0
matmul_readvariableop_resource:==-
biasadd_readvariableop_resource:=
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_717/kernel/Regularizer/Abs/ReadVariableOp¢2dense_717/kernel/Regularizer/Square/ReadVariableOpt
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
:ÿÿÿÿÿÿÿÿÿ=g
"dense_717/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_717/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:==*
dtype0
 dense_717/kernel/Regularizer/AbsAbs7dense_717/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_717/kernel/Regularizer/SumSum$dense_717/kernel/Regularizer/Abs:y:0-dense_717/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_717/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_717/kernel/Regularizer/mulMul+dense_717/kernel/Regularizer/mul/x:output:0)dense_717/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_717/kernel/Regularizer/addAddV2+dense_717/kernel/Regularizer/Const:output:0$dense_717/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_717/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:==*
dtype0
#dense_717/kernel/Regularizer/SquareSquare:dense_717/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_717/kernel/Regularizer/Sum_1Sum'dense_717/kernel/Regularizer/Square:y:0-dense_717/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_717/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_717/kernel/Regularizer/mul_1Mul-dense_717/kernel/Regularizer/mul_1/x:output:0+dense_717/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_717/kernel/Regularizer/add_1AddV2$dense_717/kernel/Regularizer/add:z:0&dense_717/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=Þ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_717/kernel/Regularizer/Abs/ReadVariableOp3^dense_717/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_717/kernel/Regularizer/Abs/ReadVariableOp/dense_717/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_717/kernel/Regularizer/Square/ReadVariableOp2dense_717/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
â
¿B
J__inference_sequential_71_layer_call_and_return_conditional_losses_1124807

inputs
normalization_71_sub_y
normalization_71_sqrt_x:
(dense_711_matmul_readvariableop_resource:i7
)dense_711_biasadd_readvariableop_resource:iM
?batch_normalization_640_assignmovingavg_readvariableop_resource:iO
Abatch_normalization_640_assignmovingavg_1_readvariableop_resource:iK
=batch_normalization_640_batchnorm_mul_readvariableop_resource:iG
9batch_normalization_640_batchnorm_readvariableop_resource:i:
(dense_712_matmul_readvariableop_resource:ii7
)dense_712_biasadd_readvariableop_resource:iM
?batch_normalization_641_assignmovingavg_readvariableop_resource:iO
Abatch_normalization_641_assignmovingavg_1_readvariableop_resource:iK
=batch_normalization_641_batchnorm_mul_readvariableop_resource:iG
9batch_normalization_641_batchnorm_readvariableop_resource:i:
(dense_713_matmul_readvariableop_resource:ii7
)dense_713_biasadd_readvariableop_resource:iM
?batch_normalization_642_assignmovingavg_readvariableop_resource:iO
Abatch_normalization_642_assignmovingavg_1_readvariableop_resource:iK
=batch_normalization_642_batchnorm_mul_readvariableop_resource:iG
9batch_normalization_642_batchnorm_readvariableop_resource:i:
(dense_714_matmul_readvariableop_resource:ii7
)dense_714_biasadd_readvariableop_resource:iM
?batch_normalization_643_assignmovingavg_readvariableop_resource:iO
Abatch_normalization_643_assignmovingavg_1_readvariableop_resource:iK
=batch_normalization_643_batchnorm_mul_readvariableop_resource:iG
9batch_normalization_643_batchnorm_readvariableop_resource:i:
(dense_715_matmul_readvariableop_resource:ii7
)dense_715_biasadd_readvariableop_resource:iM
?batch_normalization_644_assignmovingavg_readvariableop_resource:iO
Abatch_normalization_644_assignmovingavg_1_readvariableop_resource:iK
=batch_normalization_644_batchnorm_mul_readvariableop_resource:iG
9batch_normalization_644_batchnorm_readvariableop_resource:i:
(dense_716_matmul_readvariableop_resource:i=7
)dense_716_biasadd_readvariableop_resource:=M
?batch_normalization_645_assignmovingavg_readvariableop_resource:=O
Abatch_normalization_645_assignmovingavg_1_readvariableop_resource:=K
=batch_normalization_645_batchnorm_mul_readvariableop_resource:=G
9batch_normalization_645_batchnorm_readvariableop_resource:=:
(dense_717_matmul_readvariableop_resource:==7
)dense_717_biasadd_readvariableop_resource:=M
?batch_normalization_646_assignmovingavg_readvariableop_resource:=O
Abatch_normalization_646_assignmovingavg_1_readvariableop_resource:=K
=batch_normalization_646_batchnorm_mul_readvariableop_resource:=G
9batch_normalization_646_batchnorm_readvariableop_resource:=:
(dense_718_matmul_readvariableop_resource:=77
)dense_718_biasadd_readvariableop_resource:7M
?batch_normalization_647_assignmovingavg_readvariableop_resource:7O
Abatch_normalization_647_assignmovingavg_1_readvariableop_resource:7K
=batch_normalization_647_batchnorm_mul_readvariableop_resource:7G
9batch_normalization_647_batchnorm_readvariableop_resource:7:
(dense_719_matmul_readvariableop_resource:777
)dense_719_biasadd_readvariableop_resource:7M
?batch_normalization_648_assignmovingavg_readvariableop_resource:7O
Abatch_normalization_648_assignmovingavg_1_readvariableop_resource:7K
=batch_normalization_648_batchnorm_mul_readvariableop_resource:7G
9batch_normalization_648_batchnorm_readvariableop_resource:7:
(dense_720_matmul_readvariableop_resource:77
)dense_720_biasadd_readvariableop_resource:
identity¢'batch_normalization_640/AssignMovingAvg¢6batch_normalization_640/AssignMovingAvg/ReadVariableOp¢)batch_normalization_640/AssignMovingAvg_1¢8batch_normalization_640/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_640/batchnorm/ReadVariableOp¢4batch_normalization_640/batchnorm/mul/ReadVariableOp¢'batch_normalization_641/AssignMovingAvg¢6batch_normalization_641/AssignMovingAvg/ReadVariableOp¢)batch_normalization_641/AssignMovingAvg_1¢8batch_normalization_641/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_641/batchnorm/ReadVariableOp¢4batch_normalization_641/batchnorm/mul/ReadVariableOp¢'batch_normalization_642/AssignMovingAvg¢6batch_normalization_642/AssignMovingAvg/ReadVariableOp¢)batch_normalization_642/AssignMovingAvg_1¢8batch_normalization_642/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_642/batchnorm/ReadVariableOp¢4batch_normalization_642/batchnorm/mul/ReadVariableOp¢'batch_normalization_643/AssignMovingAvg¢6batch_normalization_643/AssignMovingAvg/ReadVariableOp¢)batch_normalization_643/AssignMovingAvg_1¢8batch_normalization_643/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_643/batchnorm/ReadVariableOp¢4batch_normalization_643/batchnorm/mul/ReadVariableOp¢'batch_normalization_644/AssignMovingAvg¢6batch_normalization_644/AssignMovingAvg/ReadVariableOp¢)batch_normalization_644/AssignMovingAvg_1¢8batch_normalization_644/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_644/batchnorm/ReadVariableOp¢4batch_normalization_644/batchnorm/mul/ReadVariableOp¢'batch_normalization_645/AssignMovingAvg¢6batch_normalization_645/AssignMovingAvg/ReadVariableOp¢)batch_normalization_645/AssignMovingAvg_1¢8batch_normalization_645/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_645/batchnorm/ReadVariableOp¢4batch_normalization_645/batchnorm/mul/ReadVariableOp¢'batch_normalization_646/AssignMovingAvg¢6batch_normalization_646/AssignMovingAvg/ReadVariableOp¢)batch_normalization_646/AssignMovingAvg_1¢8batch_normalization_646/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_646/batchnorm/ReadVariableOp¢4batch_normalization_646/batchnorm/mul/ReadVariableOp¢'batch_normalization_647/AssignMovingAvg¢6batch_normalization_647/AssignMovingAvg/ReadVariableOp¢)batch_normalization_647/AssignMovingAvg_1¢8batch_normalization_647/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_647/batchnorm/ReadVariableOp¢4batch_normalization_647/batchnorm/mul/ReadVariableOp¢'batch_normalization_648/AssignMovingAvg¢6batch_normalization_648/AssignMovingAvg/ReadVariableOp¢)batch_normalization_648/AssignMovingAvg_1¢8batch_normalization_648/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_648/batchnorm/ReadVariableOp¢4batch_normalization_648/batchnorm/mul/ReadVariableOp¢ dense_711/BiasAdd/ReadVariableOp¢dense_711/MatMul/ReadVariableOp¢/dense_711/kernel/Regularizer/Abs/ReadVariableOp¢2dense_711/kernel/Regularizer/Square/ReadVariableOp¢ dense_712/BiasAdd/ReadVariableOp¢dense_712/MatMul/ReadVariableOp¢/dense_712/kernel/Regularizer/Abs/ReadVariableOp¢2dense_712/kernel/Regularizer/Square/ReadVariableOp¢ dense_713/BiasAdd/ReadVariableOp¢dense_713/MatMul/ReadVariableOp¢/dense_713/kernel/Regularizer/Abs/ReadVariableOp¢2dense_713/kernel/Regularizer/Square/ReadVariableOp¢ dense_714/BiasAdd/ReadVariableOp¢dense_714/MatMul/ReadVariableOp¢/dense_714/kernel/Regularizer/Abs/ReadVariableOp¢2dense_714/kernel/Regularizer/Square/ReadVariableOp¢ dense_715/BiasAdd/ReadVariableOp¢dense_715/MatMul/ReadVariableOp¢/dense_715/kernel/Regularizer/Abs/ReadVariableOp¢2dense_715/kernel/Regularizer/Square/ReadVariableOp¢ dense_716/BiasAdd/ReadVariableOp¢dense_716/MatMul/ReadVariableOp¢/dense_716/kernel/Regularizer/Abs/ReadVariableOp¢2dense_716/kernel/Regularizer/Square/ReadVariableOp¢ dense_717/BiasAdd/ReadVariableOp¢dense_717/MatMul/ReadVariableOp¢/dense_717/kernel/Regularizer/Abs/ReadVariableOp¢2dense_717/kernel/Regularizer/Square/ReadVariableOp¢ dense_718/BiasAdd/ReadVariableOp¢dense_718/MatMul/ReadVariableOp¢/dense_718/kernel/Regularizer/Abs/ReadVariableOp¢2dense_718/kernel/Regularizer/Square/ReadVariableOp¢ dense_719/BiasAdd/ReadVariableOp¢dense_719/MatMul/ReadVariableOp¢/dense_719/kernel/Regularizer/Abs/ReadVariableOp¢2dense_719/kernel/Regularizer/Square/ReadVariableOp¢ dense_720/BiasAdd/ReadVariableOp¢dense_720/MatMul/ReadVariableOpm
normalization_71/subSubinputsnormalization_71_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_71/SqrtSqrtnormalization_71_sqrt_x*
T0*
_output_shapes

:_
normalization_71/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_71/MaximumMaximumnormalization_71/Sqrt:y:0#normalization_71/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_71/truedivRealDivnormalization_71/sub:z:0normalization_71/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_711/MatMul/ReadVariableOpReadVariableOp(dense_711_matmul_readvariableop_resource*
_output_shapes

:i*
dtype0
dense_711/MatMulMatMulnormalization_71/truediv:z:0'dense_711/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 dense_711/BiasAdd/ReadVariableOpReadVariableOp)dense_711_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
dense_711/BiasAddBiasAdddense_711/MatMul:product:0(dense_711/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
6batch_normalization_640/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_640/moments/meanMeandense_711/BiasAdd:output:0?batch_normalization_640/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(
,batch_normalization_640/moments/StopGradientStopGradient-batch_normalization_640/moments/mean:output:0*
T0*
_output_shapes

:iË
1batch_normalization_640/moments/SquaredDifferenceSquaredDifferencedense_711/BiasAdd:output:05batch_normalization_640/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
:batch_normalization_640/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_640/moments/varianceMean5batch_normalization_640/moments/SquaredDifference:z:0Cbatch_normalization_640/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(
'batch_normalization_640/moments/SqueezeSqueeze-batch_normalization_640/moments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 £
)batch_normalization_640/moments/Squeeze_1Squeeze1batch_normalization_640/moments/variance:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 r
-batch_normalization_640/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_640/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_640_assignmovingavg_readvariableop_resource*
_output_shapes
:i*
dtype0É
+batch_normalization_640/AssignMovingAvg/subSub>batch_normalization_640/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_640/moments/Squeeze:output:0*
T0*
_output_shapes
:iÀ
+batch_normalization_640/AssignMovingAvg/mulMul/batch_normalization_640/AssignMovingAvg/sub:z:06batch_normalization_640/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i
'batch_normalization_640/AssignMovingAvgAssignSubVariableOp?batch_normalization_640_assignmovingavg_readvariableop_resource/batch_normalization_640/AssignMovingAvg/mul:z:07^batch_normalization_640/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_640/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_640/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_640_assignmovingavg_1_readvariableop_resource*
_output_shapes
:i*
dtype0Ï
-batch_normalization_640/AssignMovingAvg_1/subSub@batch_normalization_640/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_640/moments/Squeeze_1:output:0*
T0*
_output_shapes
:iÆ
-batch_normalization_640/AssignMovingAvg_1/mulMul1batch_normalization_640/AssignMovingAvg_1/sub:z:08batch_normalization_640/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i
)batch_normalization_640/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_640_assignmovingavg_1_readvariableop_resource1batch_normalization_640/AssignMovingAvg_1/mul:z:09^batch_normalization_640/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_640/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_640/batchnorm/addAddV22batch_normalization_640/moments/Squeeze_1:output:00batch_normalization_640/batchnorm/add/y:output:0*
T0*
_output_shapes
:i
'batch_normalization_640/batchnorm/RsqrtRsqrt)batch_normalization_640/batchnorm/add:z:0*
T0*
_output_shapes
:i®
4batch_normalization_640/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_640_batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0¼
%batch_normalization_640/batchnorm/mulMul+batch_normalization_640/batchnorm/Rsqrt:y:0<batch_normalization_640/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:i§
'batch_normalization_640/batchnorm/mul_1Muldense_711/BiasAdd:output:0)batch_normalization_640/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi°
'batch_normalization_640/batchnorm/mul_2Mul0batch_normalization_640/moments/Squeeze:output:0)batch_normalization_640/batchnorm/mul:z:0*
T0*
_output_shapes
:i¦
0batch_normalization_640/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_640_batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0¸
%batch_normalization_640/batchnorm/subSub8batch_normalization_640/batchnorm/ReadVariableOp:value:0+batch_normalization_640/batchnorm/mul_2:z:0*
T0*
_output_shapes
:iº
'batch_normalization_640/batchnorm/add_1AddV2+batch_normalization_640/batchnorm/mul_1:z:0)batch_normalization_640/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
leaky_re_lu_640/LeakyRelu	LeakyRelu+batch_normalization_640/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>
dense_712/MatMul/ReadVariableOpReadVariableOp(dense_712_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
dense_712/MatMulMatMul'leaky_re_lu_640/LeakyRelu:activations:0'dense_712/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 dense_712/BiasAdd/ReadVariableOpReadVariableOp)dense_712_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
dense_712/BiasAddBiasAdddense_712/MatMul:product:0(dense_712/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
6batch_normalization_641/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_641/moments/meanMeandense_712/BiasAdd:output:0?batch_normalization_641/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(
,batch_normalization_641/moments/StopGradientStopGradient-batch_normalization_641/moments/mean:output:0*
T0*
_output_shapes

:iË
1batch_normalization_641/moments/SquaredDifferenceSquaredDifferencedense_712/BiasAdd:output:05batch_normalization_641/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
:batch_normalization_641/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_641/moments/varianceMean5batch_normalization_641/moments/SquaredDifference:z:0Cbatch_normalization_641/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(
'batch_normalization_641/moments/SqueezeSqueeze-batch_normalization_641/moments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 £
)batch_normalization_641/moments/Squeeze_1Squeeze1batch_normalization_641/moments/variance:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 r
-batch_normalization_641/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_641/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_641_assignmovingavg_readvariableop_resource*
_output_shapes
:i*
dtype0É
+batch_normalization_641/AssignMovingAvg/subSub>batch_normalization_641/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_641/moments/Squeeze:output:0*
T0*
_output_shapes
:iÀ
+batch_normalization_641/AssignMovingAvg/mulMul/batch_normalization_641/AssignMovingAvg/sub:z:06batch_normalization_641/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i
'batch_normalization_641/AssignMovingAvgAssignSubVariableOp?batch_normalization_641_assignmovingavg_readvariableop_resource/batch_normalization_641/AssignMovingAvg/mul:z:07^batch_normalization_641/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_641/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_641/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_641_assignmovingavg_1_readvariableop_resource*
_output_shapes
:i*
dtype0Ï
-batch_normalization_641/AssignMovingAvg_1/subSub@batch_normalization_641/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_641/moments/Squeeze_1:output:0*
T0*
_output_shapes
:iÆ
-batch_normalization_641/AssignMovingAvg_1/mulMul1batch_normalization_641/AssignMovingAvg_1/sub:z:08batch_normalization_641/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i
)batch_normalization_641/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_641_assignmovingavg_1_readvariableop_resource1batch_normalization_641/AssignMovingAvg_1/mul:z:09^batch_normalization_641/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_641/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_641/batchnorm/addAddV22batch_normalization_641/moments/Squeeze_1:output:00batch_normalization_641/batchnorm/add/y:output:0*
T0*
_output_shapes
:i
'batch_normalization_641/batchnorm/RsqrtRsqrt)batch_normalization_641/batchnorm/add:z:0*
T0*
_output_shapes
:i®
4batch_normalization_641/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_641_batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0¼
%batch_normalization_641/batchnorm/mulMul+batch_normalization_641/batchnorm/Rsqrt:y:0<batch_normalization_641/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:i§
'batch_normalization_641/batchnorm/mul_1Muldense_712/BiasAdd:output:0)batch_normalization_641/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi°
'batch_normalization_641/batchnorm/mul_2Mul0batch_normalization_641/moments/Squeeze:output:0)batch_normalization_641/batchnorm/mul:z:0*
T0*
_output_shapes
:i¦
0batch_normalization_641/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_641_batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0¸
%batch_normalization_641/batchnorm/subSub8batch_normalization_641/batchnorm/ReadVariableOp:value:0+batch_normalization_641/batchnorm/mul_2:z:0*
T0*
_output_shapes
:iº
'batch_normalization_641/batchnorm/add_1AddV2+batch_normalization_641/batchnorm/mul_1:z:0)batch_normalization_641/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
leaky_re_lu_641/LeakyRelu	LeakyRelu+batch_normalization_641/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>
dense_713/MatMul/ReadVariableOpReadVariableOp(dense_713_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
dense_713/MatMulMatMul'leaky_re_lu_641/LeakyRelu:activations:0'dense_713/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 dense_713/BiasAdd/ReadVariableOpReadVariableOp)dense_713_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
dense_713/BiasAddBiasAdddense_713/MatMul:product:0(dense_713/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
6batch_normalization_642/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_642/moments/meanMeandense_713/BiasAdd:output:0?batch_normalization_642/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(
,batch_normalization_642/moments/StopGradientStopGradient-batch_normalization_642/moments/mean:output:0*
T0*
_output_shapes

:iË
1batch_normalization_642/moments/SquaredDifferenceSquaredDifferencedense_713/BiasAdd:output:05batch_normalization_642/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
:batch_normalization_642/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_642/moments/varianceMean5batch_normalization_642/moments/SquaredDifference:z:0Cbatch_normalization_642/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(
'batch_normalization_642/moments/SqueezeSqueeze-batch_normalization_642/moments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 £
)batch_normalization_642/moments/Squeeze_1Squeeze1batch_normalization_642/moments/variance:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 r
-batch_normalization_642/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_642/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_642_assignmovingavg_readvariableop_resource*
_output_shapes
:i*
dtype0É
+batch_normalization_642/AssignMovingAvg/subSub>batch_normalization_642/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_642/moments/Squeeze:output:0*
T0*
_output_shapes
:iÀ
+batch_normalization_642/AssignMovingAvg/mulMul/batch_normalization_642/AssignMovingAvg/sub:z:06batch_normalization_642/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i
'batch_normalization_642/AssignMovingAvgAssignSubVariableOp?batch_normalization_642_assignmovingavg_readvariableop_resource/batch_normalization_642/AssignMovingAvg/mul:z:07^batch_normalization_642/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_642/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_642/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_642_assignmovingavg_1_readvariableop_resource*
_output_shapes
:i*
dtype0Ï
-batch_normalization_642/AssignMovingAvg_1/subSub@batch_normalization_642/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_642/moments/Squeeze_1:output:0*
T0*
_output_shapes
:iÆ
-batch_normalization_642/AssignMovingAvg_1/mulMul1batch_normalization_642/AssignMovingAvg_1/sub:z:08batch_normalization_642/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i
)batch_normalization_642/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_642_assignmovingavg_1_readvariableop_resource1batch_normalization_642/AssignMovingAvg_1/mul:z:09^batch_normalization_642/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_642/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_642/batchnorm/addAddV22batch_normalization_642/moments/Squeeze_1:output:00batch_normalization_642/batchnorm/add/y:output:0*
T0*
_output_shapes
:i
'batch_normalization_642/batchnorm/RsqrtRsqrt)batch_normalization_642/batchnorm/add:z:0*
T0*
_output_shapes
:i®
4batch_normalization_642/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_642_batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0¼
%batch_normalization_642/batchnorm/mulMul+batch_normalization_642/batchnorm/Rsqrt:y:0<batch_normalization_642/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:i§
'batch_normalization_642/batchnorm/mul_1Muldense_713/BiasAdd:output:0)batch_normalization_642/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi°
'batch_normalization_642/batchnorm/mul_2Mul0batch_normalization_642/moments/Squeeze:output:0)batch_normalization_642/batchnorm/mul:z:0*
T0*
_output_shapes
:i¦
0batch_normalization_642/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_642_batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0¸
%batch_normalization_642/batchnorm/subSub8batch_normalization_642/batchnorm/ReadVariableOp:value:0+batch_normalization_642/batchnorm/mul_2:z:0*
T0*
_output_shapes
:iº
'batch_normalization_642/batchnorm/add_1AddV2+batch_normalization_642/batchnorm/mul_1:z:0)batch_normalization_642/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
leaky_re_lu_642/LeakyRelu	LeakyRelu+batch_normalization_642/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>
dense_714/MatMul/ReadVariableOpReadVariableOp(dense_714_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
dense_714/MatMulMatMul'leaky_re_lu_642/LeakyRelu:activations:0'dense_714/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 dense_714/BiasAdd/ReadVariableOpReadVariableOp)dense_714_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
dense_714/BiasAddBiasAdddense_714/MatMul:product:0(dense_714/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
6batch_normalization_643/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_643/moments/meanMeandense_714/BiasAdd:output:0?batch_normalization_643/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(
,batch_normalization_643/moments/StopGradientStopGradient-batch_normalization_643/moments/mean:output:0*
T0*
_output_shapes

:iË
1batch_normalization_643/moments/SquaredDifferenceSquaredDifferencedense_714/BiasAdd:output:05batch_normalization_643/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
:batch_normalization_643/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_643/moments/varianceMean5batch_normalization_643/moments/SquaredDifference:z:0Cbatch_normalization_643/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(
'batch_normalization_643/moments/SqueezeSqueeze-batch_normalization_643/moments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 £
)batch_normalization_643/moments/Squeeze_1Squeeze1batch_normalization_643/moments/variance:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 r
-batch_normalization_643/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_643/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_643_assignmovingavg_readvariableop_resource*
_output_shapes
:i*
dtype0É
+batch_normalization_643/AssignMovingAvg/subSub>batch_normalization_643/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_643/moments/Squeeze:output:0*
T0*
_output_shapes
:iÀ
+batch_normalization_643/AssignMovingAvg/mulMul/batch_normalization_643/AssignMovingAvg/sub:z:06batch_normalization_643/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i
'batch_normalization_643/AssignMovingAvgAssignSubVariableOp?batch_normalization_643_assignmovingavg_readvariableop_resource/batch_normalization_643/AssignMovingAvg/mul:z:07^batch_normalization_643/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_643/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_643/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_643_assignmovingavg_1_readvariableop_resource*
_output_shapes
:i*
dtype0Ï
-batch_normalization_643/AssignMovingAvg_1/subSub@batch_normalization_643/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_643/moments/Squeeze_1:output:0*
T0*
_output_shapes
:iÆ
-batch_normalization_643/AssignMovingAvg_1/mulMul1batch_normalization_643/AssignMovingAvg_1/sub:z:08batch_normalization_643/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i
)batch_normalization_643/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_643_assignmovingavg_1_readvariableop_resource1batch_normalization_643/AssignMovingAvg_1/mul:z:09^batch_normalization_643/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_643/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_643/batchnorm/addAddV22batch_normalization_643/moments/Squeeze_1:output:00batch_normalization_643/batchnorm/add/y:output:0*
T0*
_output_shapes
:i
'batch_normalization_643/batchnorm/RsqrtRsqrt)batch_normalization_643/batchnorm/add:z:0*
T0*
_output_shapes
:i®
4batch_normalization_643/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_643_batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0¼
%batch_normalization_643/batchnorm/mulMul+batch_normalization_643/batchnorm/Rsqrt:y:0<batch_normalization_643/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:i§
'batch_normalization_643/batchnorm/mul_1Muldense_714/BiasAdd:output:0)batch_normalization_643/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi°
'batch_normalization_643/batchnorm/mul_2Mul0batch_normalization_643/moments/Squeeze:output:0)batch_normalization_643/batchnorm/mul:z:0*
T0*
_output_shapes
:i¦
0batch_normalization_643/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_643_batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0¸
%batch_normalization_643/batchnorm/subSub8batch_normalization_643/batchnorm/ReadVariableOp:value:0+batch_normalization_643/batchnorm/mul_2:z:0*
T0*
_output_shapes
:iº
'batch_normalization_643/batchnorm/add_1AddV2+batch_normalization_643/batchnorm/mul_1:z:0)batch_normalization_643/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
leaky_re_lu_643/LeakyRelu	LeakyRelu+batch_normalization_643/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>
dense_715/MatMul/ReadVariableOpReadVariableOp(dense_715_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
dense_715/MatMulMatMul'leaky_re_lu_643/LeakyRelu:activations:0'dense_715/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 dense_715/BiasAdd/ReadVariableOpReadVariableOp)dense_715_biasadd_readvariableop_resource*
_output_shapes
:i*
dtype0
dense_715/BiasAddBiasAdddense_715/MatMul:product:0(dense_715/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
6batch_normalization_644/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_644/moments/meanMeandense_715/BiasAdd:output:0?batch_normalization_644/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(
,batch_normalization_644/moments/StopGradientStopGradient-batch_normalization_644/moments/mean:output:0*
T0*
_output_shapes

:iË
1batch_normalization_644/moments/SquaredDifferenceSquaredDifferencedense_715/BiasAdd:output:05batch_normalization_644/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
:batch_normalization_644/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_644/moments/varianceMean5batch_normalization_644/moments/SquaredDifference:z:0Cbatch_normalization_644/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(
'batch_normalization_644/moments/SqueezeSqueeze-batch_normalization_644/moments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 £
)batch_normalization_644/moments/Squeeze_1Squeeze1batch_normalization_644/moments/variance:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 r
-batch_normalization_644/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_644/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_644_assignmovingavg_readvariableop_resource*
_output_shapes
:i*
dtype0É
+batch_normalization_644/AssignMovingAvg/subSub>batch_normalization_644/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_644/moments/Squeeze:output:0*
T0*
_output_shapes
:iÀ
+batch_normalization_644/AssignMovingAvg/mulMul/batch_normalization_644/AssignMovingAvg/sub:z:06batch_normalization_644/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i
'batch_normalization_644/AssignMovingAvgAssignSubVariableOp?batch_normalization_644_assignmovingavg_readvariableop_resource/batch_normalization_644/AssignMovingAvg/mul:z:07^batch_normalization_644/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_644/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_644/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_644_assignmovingavg_1_readvariableop_resource*
_output_shapes
:i*
dtype0Ï
-batch_normalization_644/AssignMovingAvg_1/subSub@batch_normalization_644/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_644/moments/Squeeze_1:output:0*
T0*
_output_shapes
:iÆ
-batch_normalization_644/AssignMovingAvg_1/mulMul1batch_normalization_644/AssignMovingAvg_1/sub:z:08batch_normalization_644/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i
)batch_normalization_644/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_644_assignmovingavg_1_readvariableop_resource1batch_normalization_644/AssignMovingAvg_1/mul:z:09^batch_normalization_644/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_644/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_644/batchnorm/addAddV22batch_normalization_644/moments/Squeeze_1:output:00batch_normalization_644/batchnorm/add/y:output:0*
T0*
_output_shapes
:i
'batch_normalization_644/batchnorm/RsqrtRsqrt)batch_normalization_644/batchnorm/add:z:0*
T0*
_output_shapes
:i®
4batch_normalization_644/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_644_batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0¼
%batch_normalization_644/batchnorm/mulMul+batch_normalization_644/batchnorm/Rsqrt:y:0<batch_normalization_644/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:i§
'batch_normalization_644/batchnorm/mul_1Muldense_715/BiasAdd:output:0)batch_normalization_644/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi°
'batch_normalization_644/batchnorm/mul_2Mul0batch_normalization_644/moments/Squeeze:output:0)batch_normalization_644/batchnorm/mul:z:0*
T0*
_output_shapes
:i¦
0batch_normalization_644/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_644_batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0¸
%batch_normalization_644/batchnorm/subSub8batch_normalization_644/batchnorm/ReadVariableOp:value:0+batch_normalization_644/batchnorm/mul_2:z:0*
T0*
_output_shapes
:iº
'batch_normalization_644/batchnorm/add_1AddV2+batch_normalization_644/batchnorm/mul_1:z:0)batch_normalization_644/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
leaky_re_lu_644/LeakyRelu	LeakyRelu+batch_normalization_644/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>
dense_716/MatMul/ReadVariableOpReadVariableOp(dense_716_matmul_readvariableop_resource*
_output_shapes

:i=*
dtype0
dense_716/MatMulMatMul'leaky_re_lu_644/LeakyRelu:activations:0'dense_716/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 dense_716/BiasAdd/ReadVariableOpReadVariableOp)dense_716_biasadd_readvariableop_resource*
_output_shapes
:=*
dtype0
dense_716/BiasAddBiasAdddense_716/MatMul:product:0(dense_716/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
6batch_normalization_645/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_645/moments/meanMeandense_716/BiasAdd:output:0?batch_normalization_645/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(
,batch_normalization_645/moments/StopGradientStopGradient-batch_normalization_645/moments/mean:output:0*
T0*
_output_shapes

:=Ë
1batch_normalization_645/moments/SquaredDifferenceSquaredDifferencedense_716/BiasAdd:output:05batch_normalization_645/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
:batch_normalization_645/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_645/moments/varianceMean5batch_normalization_645/moments/SquaredDifference:z:0Cbatch_normalization_645/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(
'batch_normalization_645/moments/SqueezeSqueeze-batch_normalization_645/moments/mean:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 £
)batch_normalization_645/moments/Squeeze_1Squeeze1batch_normalization_645/moments/variance:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 r
-batch_normalization_645/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_645/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_645_assignmovingavg_readvariableop_resource*
_output_shapes
:=*
dtype0É
+batch_normalization_645/AssignMovingAvg/subSub>batch_normalization_645/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_645/moments/Squeeze:output:0*
T0*
_output_shapes
:=À
+batch_normalization_645/AssignMovingAvg/mulMul/batch_normalization_645/AssignMovingAvg/sub:z:06batch_normalization_645/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:=
'batch_normalization_645/AssignMovingAvgAssignSubVariableOp?batch_normalization_645_assignmovingavg_readvariableop_resource/batch_normalization_645/AssignMovingAvg/mul:z:07^batch_normalization_645/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_645/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_645/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_645_assignmovingavg_1_readvariableop_resource*
_output_shapes
:=*
dtype0Ï
-batch_normalization_645/AssignMovingAvg_1/subSub@batch_normalization_645/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_645/moments/Squeeze_1:output:0*
T0*
_output_shapes
:=Æ
-batch_normalization_645/AssignMovingAvg_1/mulMul1batch_normalization_645/AssignMovingAvg_1/sub:z:08batch_normalization_645/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:=
)batch_normalization_645/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_645_assignmovingavg_1_readvariableop_resource1batch_normalization_645/AssignMovingAvg_1/mul:z:09^batch_normalization_645/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_645/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_645/batchnorm/addAddV22batch_normalization_645/moments/Squeeze_1:output:00batch_normalization_645/batchnorm/add/y:output:0*
T0*
_output_shapes
:=
'batch_normalization_645/batchnorm/RsqrtRsqrt)batch_normalization_645/batchnorm/add:z:0*
T0*
_output_shapes
:=®
4batch_normalization_645/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_645_batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0¼
%batch_normalization_645/batchnorm/mulMul+batch_normalization_645/batchnorm/Rsqrt:y:0<batch_normalization_645/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=§
'batch_normalization_645/batchnorm/mul_1Muldense_716/BiasAdd:output:0)batch_normalization_645/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=°
'batch_normalization_645/batchnorm/mul_2Mul0batch_normalization_645/moments/Squeeze:output:0)batch_normalization_645/batchnorm/mul:z:0*
T0*
_output_shapes
:=¦
0batch_normalization_645/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_645_batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0¸
%batch_normalization_645/batchnorm/subSub8batch_normalization_645/batchnorm/ReadVariableOp:value:0+batch_normalization_645/batchnorm/mul_2:z:0*
T0*
_output_shapes
:=º
'batch_normalization_645/batchnorm/add_1AddV2+batch_normalization_645/batchnorm/mul_1:z:0)batch_normalization_645/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
leaky_re_lu_645/LeakyRelu	LeakyRelu+batch_normalization_645/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>
dense_717/MatMul/ReadVariableOpReadVariableOp(dense_717_matmul_readvariableop_resource*
_output_shapes

:==*
dtype0
dense_717/MatMulMatMul'leaky_re_lu_645/LeakyRelu:activations:0'dense_717/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 dense_717/BiasAdd/ReadVariableOpReadVariableOp)dense_717_biasadd_readvariableop_resource*
_output_shapes
:=*
dtype0
dense_717/BiasAddBiasAdddense_717/MatMul:product:0(dense_717/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
6batch_normalization_646/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_646/moments/meanMeandense_717/BiasAdd:output:0?batch_normalization_646/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(
,batch_normalization_646/moments/StopGradientStopGradient-batch_normalization_646/moments/mean:output:0*
T0*
_output_shapes

:=Ë
1batch_normalization_646/moments/SquaredDifferenceSquaredDifferencedense_717/BiasAdd:output:05batch_normalization_646/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
:batch_normalization_646/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_646/moments/varianceMean5batch_normalization_646/moments/SquaredDifference:z:0Cbatch_normalization_646/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:=*
	keep_dims(
'batch_normalization_646/moments/SqueezeSqueeze-batch_normalization_646/moments/mean:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 £
)batch_normalization_646/moments/Squeeze_1Squeeze1batch_normalization_646/moments/variance:output:0*
T0*
_output_shapes
:=*
squeeze_dims
 r
-batch_normalization_646/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_646/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_646_assignmovingavg_readvariableop_resource*
_output_shapes
:=*
dtype0É
+batch_normalization_646/AssignMovingAvg/subSub>batch_normalization_646/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_646/moments/Squeeze:output:0*
T0*
_output_shapes
:=À
+batch_normalization_646/AssignMovingAvg/mulMul/batch_normalization_646/AssignMovingAvg/sub:z:06batch_normalization_646/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:=
'batch_normalization_646/AssignMovingAvgAssignSubVariableOp?batch_normalization_646_assignmovingavg_readvariableop_resource/batch_normalization_646/AssignMovingAvg/mul:z:07^batch_normalization_646/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_646/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_646/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_646_assignmovingavg_1_readvariableop_resource*
_output_shapes
:=*
dtype0Ï
-batch_normalization_646/AssignMovingAvg_1/subSub@batch_normalization_646/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_646/moments/Squeeze_1:output:0*
T0*
_output_shapes
:=Æ
-batch_normalization_646/AssignMovingAvg_1/mulMul1batch_normalization_646/AssignMovingAvg_1/sub:z:08batch_normalization_646/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:=
)batch_normalization_646/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_646_assignmovingavg_1_readvariableop_resource1batch_normalization_646/AssignMovingAvg_1/mul:z:09^batch_normalization_646/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_646/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_646/batchnorm/addAddV22batch_normalization_646/moments/Squeeze_1:output:00batch_normalization_646/batchnorm/add/y:output:0*
T0*
_output_shapes
:=
'batch_normalization_646/batchnorm/RsqrtRsqrt)batch_normalization_646/batchnorm/add:z:0*
T0*
_output_shapes
:=®
4batch_normalization_646/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_646_batchnorm_mul_readvariableop_resource*
_output_shapes
:=*
dtype0¼
%batch_normalization_646/batchnorm/mulMul+batch_normalization_646/batchnorm/Rsqrt:y:0<batch_normalization_646/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:=§
'batch_normalization_646/batchnorm/mul_1Muldense_717/BiasAdd:output:0)batch_normalization_646/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=°
'batch_normalization_646/batchnorm/mul_2Mul0batch_normalization_646/moments/Squeeze:output:0)batch_normalization_646/batchnorm/mul:z:0*
T0*
_output_shapes
:=¦
0batch_normalization_646/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_646_batchnorm_readvariableop_resource*
_output_shapes
:=*
dtype0¸
%batch_normalization_646/batchnorm/subSub8batch_normalization_646/batchnorm/ReadVariableOp:value:0+batch_normalization_646/batchnorm/mul_2:z:0*
T0*
_output_shapes
:=º
'batch_normalization_646/batchnorm/add_1AddV2+batch_normalization_646/batchnorm/mul_1:z:0)batch_normalization_646/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
leaky_re_lu_646/LeakyRelu	LeakyRelu+batch_normalization_646/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
alpha%>
dense_718/MatMul/ReadVariableOpReadVariableOp(dense_718_matmul_readvariableop_resource*
_output_shapes

:=7*
dtype0
dense_718/MatMulMatMul'leaky_re_lu_646/LeakyRelu:activations:0'dense_718/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 dense_718/BiasAdd/ReadVariableOpReadVariableOp)dense_718_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0
dense_718/BiasAddBiasAdddense_718/MatMul:product:0(dense_718/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
6batch_normalization_647/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_647/moments/meanMeandense_718/BiasAdd:output:0?batch_normalization_647/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(
,batch_normalization_647/moments/StopGradientStopGradient-batch_normalization_647/moments/mean:output:0*
T0*
_output_shapes

:7Ë
1batch_normalization_647/moments/SquaredDifferenceSquaredDifferencedense_718/BiasAdd:output:05batch_normalization_647/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
:batch_normalization_647/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_647/moments/varianceMean5batch_normalization_647/moments/SquaredDifference:z:0Cbatch_normalization_647/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(
'batch_normalization_647/moments/SqueezeSqueeze-batch_normalization_647/moments/mean:output:0*
T0*
_output_shapes
:7*
squeeze_dims
 £
)batch_normalization_647/moments/Squeeze_1Squeeze1batch_normalization_647/moments/variance:output:0*
T0*
_output_shapes
:7*
squeeze_dims
 r
-batch_normalization_647/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_647/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_647_assignmovingavg_readvariableop_resource*
_output_shapes
:7*
dtype0É
+batch_normalization_647/AssignMovingAvg/subSub>batch_normalization_647/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_647/moments/Squeeze:output:0*
T0*
_output_shapes
:7À
+batch_normalization_647/AssignMovingAvg/mulMul/batch_normalization_647/AssignMovingAvg/sub:z:06batch_normalization_647/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:7
'batch_normalization_647/AssignMovingAvgAssignSubVariableOp?batch_normalization_647_assignmovingavg_readvariableop_resource/batch_normalization_647/AssignMovingAvg/mul:z:07^batch_normalization_647/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_647/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_647/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_647_assignmovingavg_1_readvariableop_resource*
_output_shapes
:7*
dtype0Ï
-batch_normalization_647/AssignMovingAvg_1/subSub@batch_normalization_647/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_647/moments/Squeeze_1:output:0*
T0*
_output_shapes
:7Æ
-batch_normalization_647/AssignMovingAvg_1/mulMul1batch_normalization_647/AssignMovingAvg_1/sub:z:08batch_normalization_647/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:7
)batch_normalization_647/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_647_assignmovingavg_1_readvariableop_resource1batch_normalization_647/AssignMovingAvg_1/mul:z:09^batch_normalization_647/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_647/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_647/batchnorm/addAddV22batch_normalization_647/moments/Squeeze_1:output:00batch_normalization_647/batchnorm/add/y:output:0*
T0*
_output_shapes
:7
'batch_normalization_647/batchnorm/RsqrtRsqrt)batch_normalization_647/batchnorm/add:z:0*
T0*
_output_shapes
:7®
4batch_normalization_647/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_647_batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0¼
%batch_normalization_647/batchnorm/mulMul+batch_normalization_647/batchnorm/Rsqrt:y:0<batch_normalization_647/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7§
'batch_normalization_647/batchnorm/mul_1Muldense_718/BiasAdd:output:0)batch_normalization_647/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7°
'batch_normalization_647/batchnorm/mul_2Mul0batch_normalization_647/moments/Squeeze:output:0)batch_normalization_647/batchnorm/mul:z:0*
T0*
_output_shapes
:7¦
0batch_normalization_647/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_647_batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0¸
%batch_normalization_647/batchnorm/subSub8batch_normalization_647/batchnorm/ReadVariableOp:value:0+batch_normalization_647/batchnorm/mul_2:z:0*
T0*
_output_shapes
:7º
'batch_normalization_647/batchnorm/add_1AddV2+batch_normalization_647/batchnorm/mul_1:z:0)batch_normalization_647/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
leaky_re_lu_647/LeakyRelu	LeakyRelu+batch_normalization_647/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>
dense_719/MatMul/ReadVariableOpReadVariableOp(dense_719_matmul_readvariableop_resource*
_output_shapes

:77*
dtype0
dense_719/MatMulMatMul'leaky_re_lu_647/LeakyRelu:activations:0'dense_719/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 dense_719/BiasAdd/ReadVariableOpReadVariableOp)dense_719_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0
dense_719/BiasAddBiasAdddense_719/MatMul:product:0(dense_719/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
6batch_normalization_648/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_648/moments/meanMeandense_719/BiasAdd:output:0?batch_normalization_648/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(
,batch_normalization_648/moments/StopGradientStopGradient-batch_normalization_648/moments/mean:output:0*
T0*
_output_shapes

:7Ë
1batch_normalization_648/moments/SquaredDifferenceSquaredDifferencedense_719/BiasAdd:output:05batch_normalization_648/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
:batch_normalization_648/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_648/moments/varianceMean5batch_normalization_648/moments/SquaredDifference:z:0Cbatch_normalization_648/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(
'batch_normalization_648/moments/SqueezeSqueeze-batch_normalization_648/moments/mean:output:0*
T0*
_output_shapes
:7*
squeeze_dims
 £
)batch_normalization_648/moments/Squeeze_1Squeeze1batch_normalization_648/moments/variance:output:0*
T0*
_output_shapes
:7*
squeeze_dims
 r
-batch_normalization_648/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_648/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_648_assignmovingavg_readvariableop_resource*
_output_shapes
:7*
dtype0É
+batch_normalization_648/AssignMovingAvg/subSub>batch_normalization_648/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_648/moments/Squeeze:output:0*
T0*
_output_shapes
:7À
+batch_normalization_648/AssignMovingAvg/mulMul/batch_normalization_648/AssignMovingAvg/sub:z:06batch_normalization_648/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:7
'batch_normalization_648/AssignMovingAvgAssignSubVariableOp?batch_normalization_648_assignmovingavg_readvariableop_resource/batch_normalization_648/AssignMovingAvg/mul:z:07^batch_normalization_648/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_648/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_648/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_648_assignmovingavg_1_readvariableop_resource*
_output_shapes
:7*
dtype0Ï
-batch_normalization_648/AssignMovingAvg_1/subSub@batch_normalization_648/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_648/moments/Squeeze_1:output:0*
T0*
_output_shapes
:7Æ
-batch_normalization_648/AssignMovingAvg_1/mulMul1batch_normalization_648/AssignMovingAvg_1/sub:z:08batch_normalization_648/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:7
)batch_normalization_648/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_648_assignmovingavg_1_readvariableop_resource1batch_normalization_648/AssignMovingAvg_1/mul:z:09^batch_normalization_648/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_648/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_648/batchnorm/addAddV22batch_normalization_648/moments/Squeeze_1:output:00batch_normalization_648/batchnorm/add/y:output:0*
T0*
_output_shapes
:7
'batch_normalization_648/batchnorm/RsqrtRsqrt)batch_normalization_648/batchnorm/add:z:0*
T0*
_output_shapes
:7®
4batch_normalization_648/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_648_batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0¼
%batch_normalization_648/batchnorm/mulMul+batch_normalization_648/batchnorm/Rsqrt:y:0<batch_normalization_648/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7§
'batch_normalization_648/batchnorm/mul_1Muldense_719/BiasAdd:output:0)batch_normalization_648/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7°
'batch_normalization_648/batchnorm/mul_2Mul0batch_normalization_648/moments/Squeeze:output:0)batch_normalization_648/batchnorm/mul:z:0*
T0*
_output_shapes
:7¦
0batch_normalization_648/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_648_batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0¸
%batch_normalization_648/batchnorm/subSub8batch_normalization_648/batchnorm/ReadVariableOp:value:0+batch_normalization_648/batchnorm/mul_2:z:0*
T0*
_output_shapes
:7º
'batch_normalization_648/batchnorm/add_1AddV2+batch_normalization_648/batchnorm/mul_1:z:0)batch_normalization_648/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
leaky_re_lu_648/LeakyRelu	LeakyRelu+batch_normalization_648/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>
dense_720/MatMul/ReadVariableOpReadVariableOp(dense_720_matmul_readvariableop_resource*
_output_shapes

:7*
dtype0
dense_720/MatMulMatMul'leaky_re_lu_648/LeakyRelu:activations:0'dense_720/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_720/BiasAdd/ReadVariableOpReadVariableOp)dense_720_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_720/BiasAddBiasAdddense_720/MatMul:product:0(dense_720/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"dense_711/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_711/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_711_matmul_readvariableop_resource*
_output_shapes

:i*
dtype0
 dense_711/kernel/Regularizer/AbsAbs7dense_711/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_711/kernel/Regularizer/SumSum$dense_711/kernel/Regularizer/Abs:y:0-dense_711/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_711/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_711/kernel/Regularizer/mulMul+dense_711/kernel/Regularizer/mul/x:output:0)dense_711/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_711/kernel/Regularizer/addAddV2+dense_711/kernel/Regularizer/Const:output:0$dense_711/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_711/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_711_matmul_readvariableop_resource*
_output_shapes

:i*
dtype0
#dense_711/kernel/Regularizer/SquareSquare:dense_711/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_711/kernel/Regularizer/Sum_1Sum'dense_711/kernel/Regularizer/Square:y:0-dense_711/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_711/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_711/kernel/Regularizer/mul_1Mul-dense_711/kernel/Regularizer/mul_1/x:output:0+dense_711/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_711/kernel/Regularizer/add_1AddV2$dense_711/kernel/Regularizer/add:z:0&dense_711/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_712/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_712/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_712_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_712/kernel/Regularizer/AbsAbs7dense_712/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_712/kernel/Regularizer/SumSum$dense_712/kernel/Regularizer/Abs:y:0-dense_712/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_712/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_712/kernel/Regularizer/mulMul+dense_712/kernel/Regularizer/mul/x:output:0)dense_712/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_712/kernel/Regularizer/addAddV2+dense_712/kernel/Regularizer/Const:output:0$dense_712/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_712/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_712_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_712/kernel/Regularizer/SquareSquare:dense_712/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_712/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_712/kernel/Regularizer/Sum_1Sum'dense_712/kernel/Regularizer/Square:y:0-dense_712/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_712/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_712/kernel/Regularizer/mul_1Mul-dense_712/kernel/Regularizer/mul_1/x:output:0+dense_712/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_712/kernel/Regularizer/add_1AddV2$dense_712/kernel/Regularizer/add:z:0&dense_712/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_713/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_713/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_713_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_713/kernel/Regularizer/AbsAbs7dense_713/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_713/kernel/Regularizer/SumSum$dense_713/kernel/Regularizer/Abs:y:0-dense_713/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_713/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_713/kernel/Regularizer/mulMul+dense_713/kernel/Regularizer/mul/x:output:0)dense_713/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_713/kernel/Regularizer/addAddV2+dense_713/kernel/Regularizer/Const:output:0$dense_713/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_713/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_713_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_713/kernel/Regularizer/SquareSquare:dense_713/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_713/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_713/kernel/Regularizer/Sum_1Sum'dense_713/kernel/Regularizer/Square:y:0-dense_713/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_713/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_713/kernel/Regularizer/mul_1Mul-dense_713/kernel/Regularizer/mul_1/x:output:0+dense_713/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_713/kernel/Regularizer/add_1AddV2$dense_713/kernel/Regularizer/add:z:0&dense_713/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_714/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_714/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_714_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_714/kernel/Regularizer/AbsAbs7dense_714/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_714/kernel/Regularizer/SumSum$dense_714/kernel/Regularizer/Abs:y:0-dense_714/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_714/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_714/kernel/Regularizer/mulMul+dense_714/kernel/Regularizer/mul/x:output:0)dense_714/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_714/kernel/Regularizer/addAddV2+dense_714/kernel/Regularizer/Const:output:0$dense_714/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_714/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_714_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_714/kernel/Regularizer/SquareSquare:dense_714/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_714/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_714/kernel/Regularizer/Sum_1Sum'dense_714/kernel/Regularizer/Square:y:0-dense_714/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_714/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_714/kernel/Regularizer/mul_1Mul-dense_714/kernel/Regularizer/mul_1/x:output:0+dense_714/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_714/kernel/Regularizer/add_1AddV2$dense_714/kernel/Regularizer/add:z:0&dense_714/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_715/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_715/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_715_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
 dense_715/kernel/Regularizer/AbsAbs7dense_715/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_715/kernel/Regularizer/SumSum$dense_715/kernel/Regularizer/Abs:y:0-dense_715/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_715/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_715/kernel/Regularizer/mulMul+dense_715/kernel/Regularizer/mul/x:output:0)dense_715/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_715/kernel/Regularizer/addAddV2+dense_715/kernel/Regularizer/Const:output:0$dense_715/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_715/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_715_matmul_readvariableop_resource*
_output_shapes

:ii*
dtype0
#dense_715/kernel/Regularizer/SquareSquare:dense_715/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iiu
$dense_715/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_715/kernel/Regularizer/Sum_1Sum'dense_715/kernel/Regularizer/Square:y:0-dense_715/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_715/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_715/kernel/Regularizer/mul_1Mul-dense_715/kernel/Regularizer/mul_1/x:output:0+dense_715/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_715/kernel/Regularizer/add_1AddV2$dense_715/kernel/Regularizer/add:z:0&dense_715/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_716/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_716/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_716_matmul_readvariableop_resource*
_output_shapes

:i=*
dtype0
 dense_716/kernel/Regularizer/AbsAbs7dense_716/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_716/kernel/Regularizer/SumSum$dense_716/kernel/Regularizer/Abs:y:0-dense_716/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_716/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_716/kernel/Regularizer/mulMul+dense_716/kernel/Regularizer/mul/x:output:0)dense_716/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_716/kernel/Regularizer/addAddV2+dense_716/kernel/Regularizer/Const:output:0$dense_716/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_716/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_716_matmul_readvariableop_resource*
_output_shapes

:i=*
dtype0
#dense_716/kernel/Regularizer/SquareSquare:dense_716/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_716/kernel/Regularizer/Sum_1Sum'dense_716/kernel/Regularizer/Square:y:0-dense_716/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_716/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_716/kernel/Regularizer/mul_1Mul-dense_716/kernel/Regularizer/mul_1/x:output:0+dense_716/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_716/kernel/Regularizer/add_1AddV2$dense_716/kernel/Regularizer/add:z:0&dense_716/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_717/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_717/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_717_matmul_readvariableop_resource*
_output_shapes

:==*
dtype0
 dense_717/kernel/Regularizer/AbsAbs7dense_717/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_717/kernel/Regularizer/SumSum$dense_717/kernel/Regularizer/Abs:y:0-dense_717/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_717/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_717/kernel/Regularizer/mulMul+dense_717/kernel/Regularizer/mul/x:output:0)dense_717/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_717/kernel/Regularizer/addAddV2+dense_717/kernel/Regularizer/Const:output:0$dense_717/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_717/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_717_matmul_readvariableop_resource*
_output_shapes

:==*
dtype0
#dense_717/kernel/Regularizer/SquareSquare:dense_717/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_717/kernel/Regularizer/Sum_1Sum'dense_717/kernel/Regularizer/Square:y:0-dense_717/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_717/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_717/kernel/Regularizer/mul_1Mul-dense_717/kernel/Regularizer/mul_1/x:output:0+dense_717/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_717/kernel/Regularizer/add_1AddV2$dense_717/kernel/Regularizer/add:z:0&dense_717/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_718/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_718/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_718_matmul_readvariableop_resource*
_output_shapes

:=7*
dtype0
 dense_718/kernel/Regularizer/AbsAbs7dense_718/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_718/kernel/Regularizer/SumSum$dense_718/kernel/Regularizer/Abs:y:0-dense_718/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_718/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_718/kernel/Regularizer/mulMul+dense_718/kernel/Regularizer/mul/x:output:0)dense_718/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_718/kernel/Regularizer/addAddV2+dense_718/kernel/Regularizer/Const:output:0$dense_718/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_718/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_718_matmul_readvariableop_resource*
_output_shapes

:=7*
dtype0
#dense_718/kernel/Regularizer/SquareSquare:dense_718/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_718/kernel/Regularizer/Sum_1Sum'dense_718/kernel/Regularizer/Square:y:0-dense_718/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_718/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_718/kernel/Regularizer/mul_1Mul-dense_718/kernel/Regularizer/mul_1/x:output:0+dense_718/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_718/kernel/Regularizer/add_1AddV2$dense_718/kernel/Regularizer/add:z:0&dense_718/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"dense_719/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_719/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_719_matmul_readvariableop_resource*
_output_shapes

:77*
dtype0
 dense_719/kernel/Regularizer/AbsAbs7dense_719/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_719/kernel/Regularizer/SumSum$dense_719/kernel/Regularizer/Abs:y:0-dense_719/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_719/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_719/kernel/Regularizer/mulMul+dense_719/kernel/Regularizer/mul/x:output:0)dense_719/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_719/kernel/Regularizer/addAddV2+dense_719/kernel/Regularizer/Const:output:0$dense_719/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_719/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_719_matmul_readvariableop_resource*
_output_shapes

:77*
dtype0
#dense_719/kernel/Regularizer/SquareSquare:dense_719/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:77u
$dense_719/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_719/kernel/Regularizer/Sum_1Sum'dense_719/kernel/Regularizer/Square:y:0-dense_719/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_719/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_719/kernel/Regularizer/mul_1Mul-dense_719/kernel/Regularizer/mul_1/x:output:0+dense_719/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_719/kernel/Regularizer/add_1AddV2$dense_719/kernel/Regularizer/add:z:0&dense_719/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentitydense_720/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿë"
NoOpNoOp(^batch_normalization_640/AssignMovingAvg7^batch_normalization_640/AssignMovingAvg/ReadVariableOp*^batch_normalization_640/AssignMovingAvg_19^batch_normalization_640/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_640/batchnorm/ReadVariableOp5^batch_normalization_640/batchnorm/mul/ReadVariableOp(^batch_normalization_641/AssignMovingAvg7^batch_normalization_641/AssignMovingAvg/ReadVariableOp*^batch_normalization_641/AssignMovingAvg_19^batch_normalization_641/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_641/batchnorm/ReadVariableOp5^batch_normalization_641/batchnorm/mul/ReadVariableOp(^batch_normalization_642/AssignMovingAvg7^batch_normalization_642/AssignMovingAvg/ReadVariableOp*^batch_normalization_642/AssignMovingAvg_19^batch_normalization_642/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_642/batchnorm/ReadVariableOp5^batch_normalization_642/batchnorm/mul/ReadVariableOp(^batch_normalization_643/AssignMovingAvg7^batch_normalization_643/AssignMovingAvg/ReadVariableOp*^batch_normalization_643/AssignMovingAvg_19^batch_normalization_643/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_643/batchnorm/ReadVariableOp5^batch_normalization_643/batchnorm/mul/ReadVariableOp(^batch_normalization_644/AssignMovingAvg7^batch_normalization_644/AssignMovingAvg/ReadVariableOp*^batch_normalization_644/AssignMovingAvg_19^batch_normalization_644/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_644/batchnorm/ReadVariableOp5^batch_normalization_644/batchnorm/mul/ReadVariableOp(^batch_normalization_645/AssignMovingAvg7^batch_normalization_645/AssignMovingAvg/ReadVariableOp*^batch_normalization_645/AssignMovingAvg_19^batch_normalization_645/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_645/batchnorm/ReadVariableOp5^batch_normalization_645/batchnorm/mul/ReadVariableOp(^batch_normalization_646/AssignMovingAvg7^batch_normalization_646/AssignMovingAvg/ReadVariableOp*^batch_normalization_646/AssignMovingAvg_19^batch_normalization_646/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_646/batchnorm/ReadVariableOp5^batch_normalization_646/batchnorm/mul/ReadVariableOp(^batch_normalization_647/AssignMovingAvg7^batch_normalization_647/AssignMovingAvg/ReadVariableOp*^batch_normalization_647/AssignMovingAvg_19^batch_normalization_647/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_647/batchnorm/ReadVariableOp5^batch_normalization_647/batchnorm/mul/ReadVariableOp(^batch_normalization_648/AssignMovingAvg7^batch_normalization_648/AssignMovingAvg/ReadVariableOp*^batch_normalization_648/AssignMovingAvg_19^batch_normalization_648/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_648/batchnorm/ReadVariableOp5^batch_normalization_648/batchnorm/mul/ReadVariableOp!^dense_711/BiasAdd/ReadVariableOp ^dense_711/MatMul/ReadVariableOp0^dense_711/kernel/Regularizer/Abs/ReadVariableOp3^dense_711/kernel/Regularizer/Square/ReadVariableOp!^dense_712/BiasAdd/ReadVariableOp ^dense_712/MatMul/ReadVariableOp0^dense_712/kernel/Regularizer/Abs/ReadVariableOp3^dense_712/kernel/Regularizer/Square/ReadVariableOp!^dense_713/BiasAdd/ReadVariableOp ^dense_713/MatMul/ReadVariableOp0^dense_713/kernel/Regularizer/Abs/ReadVariableOp3^dense_713/kernel/Regularizer/Square/ReadVariableOp!^dense_714/BiasAdd/ReadVariableOp ^dense_714/MatMul/ReadVariableOp0^dense_714/kernel/Regularizer/Abs/ReadVariableOp3^dense_714/kernel/Regularizer/Square/ReadVariableOp!^dense_715/BiasAdd/ReadVariableOp ^dense_715/MatMul/ReadVariableOp0^dense_715/kernel/Regularizer/Abs/ReadVariableOp3^dense_715/kernel/Regularizer/Square/ReadVariableOp!^dense_716/BiasAdd/ReadVariableOp ^dense_716/MatMul/ReadVariableOp0^dense_716/kernel/Regularizer/Abs/ReadVariableOp3^dense_716/kernel/Regularizer/Square/ReadVariableOp!^dense_717/BiasAdd/ReadVariableOp ^dense_717/MatMul/ReadVariableOp0^dense_717/kernel/Regularizer/Abs/ReadVariableOp3^dense_717/kernel/Regularizer/Square/ReadVariableOp!^dense_718/BiasAdd/ReadVariableOp ^dense_718/MatMul/ReadVariableOp0^dense_718/kernel/Regularizer/Abs/ReadVariableOp3^dense_718/kernel/Regularizer/Square/ReadVariableOp!^dense_719/BiasAdd/ReadVariableOp ^dense_719/MatMul/ReadVariableOp0^dense_719/kernel/Regularizer/Abs/ReadVariableOp3^dense_719/kernel/Regularizer/Square/ReadVariableOp!^dense_720/BiasAdd/ReadVariableOp ^dense_720/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization_640/AssignMovingAvg'batch_normalization_640/AssignMovingAvg2p
6batch_normalization_640/AssignMovingAvg/ReadVariableOp6batch_normalization_640/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_640/AssignMovingAvg_1)batch_normalization_640/AssignMovingAvg_12t
8batch_normalization_640/AssignMovingAvg_1/ReadVariableOp8batch_normalization_640/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_640/batchnorm/ReadVariableOp0batch_normalization_640/batchnorm/ReadVariableOp2l
4batch_normalization_640/batchnorm/mul/ReadVariableOp4batch_normalization_640/batchnorm/mul/ReadVariableOp2R
'batch_normalization_641/AssignMovingAvg'batch_normalization_641/AssignMovingAvg2p
6batch_normalization_641/AssignMovingAvg/ReadVariableOp6batch_normalization_641/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_641/AssignMovingAvg_1)batch_normalization_641/AssignMovingAvg_12t
8batch_normalization_641/AssignMovingAvg_1/ReadVariableOp8batch_normalization_641/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_641/batchnorm/ReadVariableOp0batch_normalization_641/batchnorm/ReadVariableOp2l
4batch_normalization_641/batchnorm/mul/ReadVariableOp4batch_normalization_641/batchnorm/mul/ReadVariableOp2R
'batch_normalization_642/AssignMovingAvg'batch_normalization_642/AssignMovingAvg2p
6batch_normalization_642/AssignMovingAvg/ReadVariableOp6batch_normalization_642/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_642/AssignMovingAvg_1)batch_normalization_642/AssignMovingAvg_12t
8batch_normalization_642/AssignMovingAvg_1/ReadVariableOp8batch_normalization_642/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_642/batchnorm/ReadVariableOp0batch_normalization_642/batchnorm/ReadVariableOp2l
4batch_normalization_642/batchnorm/mul/ReadVariableOp4batch_normalization_642/batchnorm/mul/ReadVariableOp2R
'batch_normalization_643/AssignMovingAvg'batch_normalization_643/AssignMovingAvg2p
6batch_normalization_643/AssignMovingAvg/ReadVariableOp6batch_normalization_643/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_643/AssignMovingAvg_1)batch_normalization_643/AssignMovingAvg_12t
8batch_normalization_643/AssignMovingAvg_1/ReadVariableOp8batch_normalization_643/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_643/batchnorm/ReadVariableOp0batch_normalization_643/batchnorm/ReadVariableOp2l
4batch_normalization_643/batchnorm/mul/ReadVariableOp4batch_normalization_643/batchnorm/mul/ReadVariableOp2R
'batch_normalization_644/AssignMovingAvg'batch_normalization_644/AssignMovingAvg2p
6batch_normalization_644/AssignMovingAvg/ReadVariableOp6batch_normalization_644/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_644/AssignMovingAvg_1)batch_normalization_644/AssignMovingAvg_12t
8batch_normalization_644/AssignMovingAvg_1/ReadVariableOp8batch_normalization_644/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_644/batchnorm/ReadVariableOp0batch_normalization_644/batchnorm/ReadVariableOp2l
4batch_normalization_644/batchnorm/mul/ReadVariableOp4batch_normalization_644/batchnorm/mul/ReadVariableOp2R
'batch_normalization_645/AssignMovingAvg'batch_normalization_645/AssignMovingAvg2p
6batch_normalization_645/AssignMovingAvg/ReadVariableOp6batch_normalization_645/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_645/AssignMovingAvg_1)batch_normalization_645/AssignMovingAvg_12t
8batch_normalization_645/AssignMovingAvg_1/ReadVariableOp8batch_normalization_645/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_645/batchnorm/ReadVariableOp0batch_normalization_645/batchnorm/ReadVariableOp2l
4batch_normalization_645/batchnorm/mul/ReadVariableOp4batch_normalization_645/batchnorm/mul/ReadVariableOp2R
'batch_normalization_646/AssignMovingAvg'batch_normalization_646/AssignMovingAvg2p
6batch_normalization_646/AssignMovingAvg/ReadVariableOp6batch_normalization_646/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_646/AssignMovingAvg_1)batch_normalization_646/AssignMovingAvg_12t
8batch_normalization_646/AssignMovingAvg_1/ReadVariableOp8batch_normalization_646/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_646/batchnorm/ReadVariableOp0batch_normalization_646/batchnorm/ReadVariableOp2l
4batch_normalization_646/batchnorm/mul/ReadVariableOp4batch_normalization_646/batchnorm/mul/ReadVariableOp2R
'batch_normalization_647/AssignMovingAvg'batch_normalization_647/AssignMovingAvg2p
6batch_normalization_647/AssignMovingAvg/ReadVariableOp6batch_normalization_647/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_647/AssignMovingAvg_1)batch_normalization_647/AssignMovingAvg_12t
8batch_normalization_647/AssignMovingAvg_1/ReadVariableOp8batch_normalization_647/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_647/batchnorm/ReadVariableOp0batch_normalization_647/batchnorm/ReadVariableOp2l
4batch_normalization_647/batchnorm/mul/ReadVariableOp4batch_normalization_647/batchnorm/mul/ReadVariableOp2R
'batch_normalization_648/AssignMovingAvg'batch_normalization_648/AssignMovingAvg2p
6batch_normalization_648/AssignMovingAvg/ReadVariableOp6batch_normalization_648/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_648/AssignMovingAvg_1)batch_normalization_648/AssignMovingAvg_12t
8batch_normalization_648/AssignMovingAvg_1/ReadVariableOp8batch_normalization_648/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_648/batchnorm/ReadVariableOp0batch_normalization_648/batchnorm/ReadVariableOp2l
4batch_normalization_648/batchnorm/mul/ReadVariableOp4batch_normalization_648/batchnorm/mul/ReadVariableOp2D
 dense_711/BiasAdd/ReadVariableOp dense_711/BiasAdd/ReadVariableOp2B
dense_711/MatMul/ReadVariableOpdense_711/MatMul/ReadVariableOp2b
/dense_711/kernel/Regularizer/Abs/ReadVariableOp/dense_711/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_711/kernel/Regularizer/Square/ReadVariableOp2dense_711/kernel/Regularizer/Square/ReadVariableOp2D
 dense_712/BiasAdd/ReadVariableOp dense_712/BiasAdd/ReadVariableOp2B
dense_712/MatMul/ReadVariableOpdense_712/MatMul/ReadVariableOp2b
/dense_712/kernel/Regularizer/Abs/ReadVariableOp/dense_712/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_712/kernel/Regularizer/Square/ReadVariableOp2dense_712/kernel/Regularizer/Square/ReadVariableOp2D
 dense_713/BiasAdd/ReadVariableOp dense_713/BiasAdd/ReadVariableOp2B
dense_713/MatMul/ReadVariableOpdense_713/MatMul/ReadVariableOp2b
/dense_713/kernel/Regularizer/Abs/ReadVariableOp/dense_713/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_713/kernel/Regularizer/Square/ReadVariableOp2dense_713/kernel/Regularizer/Square/ReadVariableOp2D
 dense_714/BiasAdd/ReadVariableOp dense_714/BiasAdd/ReadVariableOp2B
dense_714/MatMul/ReadVariableOpdense_714/MatMul/ReadVariableOp2b
/dense_714/kernel/Regularizer/Abs/ReadVariableOp/dense_714/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_714/kernel/Regularizer/Square/ReadVariableOp2dense_714/kernel/Regularizer/Square/ReadVariableOp2D
 dense_715/BiasAdd/ReadVariableOp dense_715/BiasAdd/ReadVariableOp2B
dense_715/MatMul/ReadVariableOpdense_715/MatMul/ReadVariableOp2b
/dense_715/kernel/Regularizer/Abs/ReadVariableOp/dense_715/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_715/kernel/Regularizer/Square/ReadVariableOp2dense_715/kernel/Regularizer/Square/ReadVariableOp2D
 dense_716/BiasAdd/ReadVariableOp dense_716/BiasAdd/ReadVariableOp2B
dense_716/MatMul/ReadVariableOpdense_716/MatMul/ReadVariableOp2b
/dense_716/kernel/Regularizer/Abs/ReadVariableOp/dense_716/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_716/kernel/Regularizer/Square/ReadVariableOp2dense_716/kernel/Regularizer/Square/ReadVariableOp2D
 dense_717/BiasAdd/ReadVariableOp dense_717/BiasAdd/ReadVariableOp2B
dense_717/MatMul/ReadVariableOpdense_717/MatMul/ReadVariableOp2b
/dense_717/kernel/Regularizer/Abs/ReadVariableOp/dense_717/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_717/kernel/Regularizer/Square/ReadVariableOp2dense_717/kernel/Regularizer/Square/ReadVariableOp2D
 dense_718/BiasAdd/ReadVariableOp dense_718/BiasAdd/ReadVariableOp2B
dense_718/MatMul/ReadVariableOpdense_718/MatMul/ReadVariableOp2b
/dense_718/kernel/Regularizer/Abs/ReadVariableOp/dense_718/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_718/kernel/Regularizer/Square/ReadVariableOp2dense_718/kernel/Regularizer/Square/ReadVariableOp2D
 dense_719/BiasAdd/ReadVariableOp dense_719/BiasAdd/ReadVariableOp2B
dense_719/MatMul/ReadVariableOpdense_719/MatMul/ReadVariableOp2b
/dense_719/kernel/Regularizer/Abs/ReadVariableOp/dense_719/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_719/kernel/Regularizer/Square/ReadVariableOp2dense_719/kernel/Regularizer/Square/ReadVariableOp2D
 dense_720/BiasAdd/ReadVariableOp dense_720/BiasAdd/ReadVariableOp2B
dense_720/MatMul/ReadVariableOpdense_720/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ù
Ó
/__inference_sequential_71_layer_call_fn_1123842

inputs
unknown
	unknown_0
	unknown_1:i
	unknown_2:i
	unknown_3:i
	unknown_4:i
	unknown_5:i
	unknown_6:i
	unknown_7:ii
	unknown_8:i
	unknown_9:i

unknown_10:i

unknown_11:i

unknown_12:i

unknown_13:ii

unknown_14:i

unknown_15:i

unknown_16:i

unknown_17:i

unknown_18:i

unknown_19:ii

unknown_20:i

unknown_21:i

unknown_22:i

unknown_23:i

unknown_24:i

unknown_25:ii

unknown_26:i

unknown_27:i

unknown_28:i

unknown_29:i

unknown_30:i

unknown_31:i=

unknown_32:=

unknown_33:=

unknown_34:=

unknown_35:=

unknown_36:=

unknown_37:==

unknown_38:=

unknown_39:=

unknown_40:=

unknown_41:=

unknown_42:=

unknown_43:=7

unknown_44:7

unknown_45:7

unknown_46:7

unknown_47:7

unknown_48:7

unknown_49:77

unknown_50:7

unknown_51:7

unknown_52:7

unknown_53:7

unknown_54:7

unknown_55:7

unknown_56:
identity¢StatefulPartitionedCallä
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
GPU 2J 8 *S
fNRL
J__inference_sequential_71_layer_call_and_return_conditional_losses_1122088o
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
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ñ
³
T__inference_batch_normalization_640_layer_call_and_return_conditional_losses_1125072

inputs/
!batchnorm_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i1
#batchnorm_readvariableop_1_resource:i1
#batchnorm_readvariableop_2_resource:i
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:iz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_643_layer_call_fn_1125528

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
:ÿÿÿÿÿÿÿÿÿi* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_643_layer_call_and_return_conditional_losses_1121699`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿi:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs

ã
__inference_loss_fn_0_1126267J
8dense_711_kernel_regularizer_abs_readvariableop_resource:i
identity¢/dense_711/kernel/Regularizer/Abs/ReadVariableOp¢2dense_711/kernel/Regularizer/Square/ReadVariableOpg
"dense_711/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
/dense_711/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8dense_711_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:i*
dtype0
 dense_711/kernel/Regularizer/AbsAbs7dense_711/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_711/kernel/Regularizer/SumSum$dense_711/kernel/Regularizer/Abs:y:0-dense_711/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_711/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_711/kernel/Regularizer/mulMul+dense_711/kernel/Regularizer/mul/x:output:0)dense_711/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_711/kernel/Regularizer/addAddV2+dense_711/kernel/Regularizer/Const:output:0$dense_711/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: «
2dense_711/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense_711_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:i*
dtype0
#dense_711/kernel/Regularizer/SquareSquare:dense_711/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_711/kernel/Regularizer/Sum_1Sum'dense_711/kernel/Regularizer/Square:y:0-dense_711/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_711/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_711/kernel/Regularizer/mul_1Mul-dense_711/kernel/Regularizer/mul_1/x:output:0+dense_711/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_711/kernel/Regularizer/add_1AddV2$dense_711/kernel/Regularizer/add:z:0&dense_711/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
IdentityIdentity&dense_711/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ­
NoOpNoOp0^dense_711/kernel/Regularizer/Abs/ReadVariableOp3^dense_711/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_711/kernel/Regularizer/Abs/ReadVariableOp/dense_711/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_711/kernel/Regularizer/Square/ReadVariableOp2dense_711/kernel/Regularizer/Square/ReadVariableOp
æ
h
L__inference_leaky_re_lu_642_layer_call_and_return_conditional_losses_1121652

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿi:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_645_layer_call_and_return_conditional_losses_1125767

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
æ
h
L__inference_leaky_re_lu_640_layer_call_and_return_conditional_losses_1125116

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿi:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_641_layer_call_fn_1125178

inputs
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_641_layer_call_and_return_conditional_losses_1120867o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_648_layer_call_and_return_conditional_losses_1121441

inputs/
!batchnorm_readvariableop_resource:73
%batchnorm_mul_readvariableop_resource:71
#batchnorm_readvariableop_1_resource:71
#batchnorm_readvariableop_2_resource:7
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:7*
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
:7P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:7~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:7*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:7z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:7*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:7r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_646_layer_call_fn_1125886

inputs
unknown:=
	unknown_0:=
	unknown_1:=
	unknown_2:=
identity¢StatefulPartitionedCall
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
GPU 2J 8 *]
fXRV
T__inference_batch_normalization_646_layer_call_and_return_conditional_losses_1121324o
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
Ñ
³
T__inference_batch_normalization_644_layer_call_and_return_conditional_losses_1121113

inputs/
!batchnorm_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i1
#batchnorm_readvariableop_1_resource:i1
#batchnorm_readvariableop_2_resource:i
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:iz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_646_layer_call_and_return_conditional_losses_1125940

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

ã
/__inference_sequential_71_layer_call_fn_1122207
normalization_71_input
unknown
	unknown_0
	unknown_1:i
	unknown_2:i
	unknown_3:i
	unknown_4:i
	unknown_5:i
	unknown_6:i
	unknown_7:ii
	unknown_8:i
	unknown_9:i

unknown_10:i

unknown_11:i

unknown_12:i

unknown_13:ii

unknown_14:i

unknown_15:i

unknown_16:i

unknown_17:i

unknown_18:i

unknown_19:ii

unknown_20:i

unknown_21:i

unknown_22:i

unknown_23:i

unknown_24:i

unknown_25:ii

unknown_26:i

unknown_27:i

unknown_28:i

unknown_29:i

unknown_30:i

unknown_31:i=

unknown_32:=

unknown_33:=

unknown_34:=

unknown_35:=

unknown_36:=

unknown_37:==

unknown_38:=

unknown_39:=

unknown_40:=

unknown_41:=

unknown_42:=

unknown_43:=7

unknown_44:7

unknown_45:7

unknown_46:7

unknown_47:7

unknown_48:7

unknown_49:77

unknown_50:7

unknown_51:7

unknown_52:7

unknown_53:7

unknown_54:7

unknown_55:7

unknown_56:
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallnormalization_71_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8 *S
fNRL
J__inference_sequential_71_layer_call_and_return_conditional_losses_1122088o
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
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_71_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ñ
³
T__inference_batch_normalization_642_layer_call_and_return_conditional_losses_1125350

inputs/
!batchnorm_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i1
#batchnorm_readvariableop_1_resource:i1
#batchnorm_readvariableop_2_resource:i
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:i*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:iz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:i*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
¥
Þ
F__inference_dense_711_layer_call_and_return_conditional_losses_1125026

inputs0
matmul_readvariableop_resource:i-
biasadd_readvariableop_resource:i
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_711/kernel/Regularizer/Abs/ReadVariableOp¢2dense_711/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:i*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿir
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:i*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿig
"dense_711/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_711/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:i*
dtype0
 dense_711/kernel/Regularizer/AbsAbs7dense_711/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_711/kernel/Regularizer/SumSum$dense_711/kernel/Regularizer/Abs:y:0-dense_711/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_711/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *©
Ç= 
 dense_711/kernel/Regularizer/mulMul+dense_711/kernel/Regularizer/mul/x:output:0)dense_711/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_711/kernel/Regularizer/addAddV2+dense_711/kernel/Regularizer/Const:output:0$dense_711/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_711/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:i*
dtype0
#dense_711/kernel/Regularizer/SquareSquare:dense_711/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:iu
$dense_711/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_711/kernel/Regularizer/Sum_1Sum'dense_711/kernel/Regularizer/Square:y:0-dense_711/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_711/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *	À=¦
"dense_711/kernel/Regularizer/mul_1Mul-dense_711/kernel/Regularizer/mul_1/x:output:0+dense_711/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_711/kernel/Regularizer/add_1AddV2$dense_711/kernel/Regularizer/add:z:0&dense_711/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiÞ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_711/kernel/Regularizer/Abs/ReadVariableOp3^dense_711/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_711/kernel/Regularizer/Abs/ReadVariableOp/dense_711/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_711/kernel/Regularizer/Square/ReadVariableOp2dense_711/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
Þ
F__inference_dense_717_layer_call_and_return_conditional_losses_1125860

inputs0
matmul_readvariableop_resource:==-
biasadd_readvariableop_resource:=
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_717/kernel/Regularizer/Abs/ReadVariableOp¢2dense_717/kernel/Regularizer/Square/ReadVariableOpt
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
:ÿÿÿÿÿÿÿÿÿ=g
"dense_717/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_717/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:==*
dtype0
 dense_717/kernel/Regularizer/AbsAbs7dense_717/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_717/kernel/Regularizer/SumSum$dense_717/kernel/Regularizer/Abs:y:0-dense_717/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_717/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_717/kernel/Regularizer/mulMul+dense_717/kernel/Regularizer/mul/x:output:0)dense_717/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_717/kernel/Regularizer/addAddV2+dense_717/kernel/Regularizer/Const:output:0$dense_717/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_717/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:==*
dtype0
#dense_717/kernel/Regularizer/SquareSquare:dense_717/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:==u
$dense_717/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_717/kernel/Regularizer/Sum_1Sum'dense_717/kernel/Regularizer/Square:y:0-dense_717/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_717/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_717/kernel/Regularizer/mul_1Mul-dense_717/kernel/Regularizer/mul_1/x:output:0+dense_717/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_717/kernel/Regularizer/add_1AddV2$dense_717/kernel/Regularizer/add:z:0&dense_717/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=Þ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_717/kernel/Regularizer/Abs/ReadVariableOp3^dense_717/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_717/kernel/Regularizer/Abs/ReadVariableOp/dense_717/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_717/kernel/Regularizer/Square/ReadVariableOp2dense_717/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
·¾
Î^
#__inference__traced_restore_1127308
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 5
#assignvariableop_3_dense_711_kernel:i/
!assignvariableop_4_dense_711_bias:i>
0assignvariableop_5_batch_normalization_640_gamma:i=
/assignvariableop_6_batch_normalization_640_beta:iD
6assignvariableop_7_batch_normalization_640_moving_mean:iH
:assignvariableop_8_batch_normalization_640_moving_variance:i5
#assignvariableop_9_dense_712_kernel:ii0
"assignvariableop_10_dense_712_bias:i?
1assignvariableop_11_batch_normalization_641_gamma:i>
0assignvariableop_12_batch_normalization_641_beta:iE
7assignvariableop_13_batch_normalization_641_moving_mean:iI
;assignvariableop_14_batch_normalization_641_moving_variance:i6
$assignvariableop_15_dense_713_kernel:ii0
"assignvariableop_16_dense_713_bias:i?
1assignvariableop_17_batch_normalization_642_gamma:i>
0assignvariableop_18_batch_normalization_642_beta:iE
7assignvariableop_19_batch_normalization_642_moving_mean:iI
;assignvariableop_20_batch_normalization_642_moving_variance:i6
$assignvariableop_21_dense_714_kernel:ii0
"assignvariableop_22_dense_714_bias:i?
1assignvariableop_23_batch_normalization_643_gamma:i>
0assignvariableop_24_batch_normalization_643_beta:iE
7assignvariableop_25_batch_normalization_643_moving_mean:iI
;assignvariableop_26_batch_normalization_643_moving_variance:i6
$assignvariableop_27_dense_715_kernel:ii0
"assignvariableop_28_dense_715_bias:i?
1assignvariableop_29_batch_normalization_644_gamma:i>
0assignvariableop_30_batch_normalization_644_beta:iE
7assignvariableop_31_batch_normalization_644_moving_mean:iI
;assignvariableop_32_batch_normalization_644_moving_variance:i6
$assignvariableop_33_dense_716_kernel:i=0
"assignvariableop_34_dense_716_bias:=?
1assignvariableop_35_batch_normalization_645_gamma:=>
0assignvariableop_36_batch_normalization_645_beta:=E
7assignvariableop_37_batch_normalization_645_moving_mean:=I
;assignvariableop_38_batch_normalization_645_moving_variance:=6
$assignvariableop_39_dense_717_kernel:==0
"assignvariableop_40_dense_717_bias:=?
1assignvariableop_41_batch_normalization_646_gamma:=>
0assignvariableop_42_batch_normalization_646_beta:=E
7assignvariableop_43_batch_normalization_646_moving_mean:=I
;assignvariableop_44_batch_normalization_646_moving_variance:=6
$assignvariableop_45_dense_718_kernel:=70
"assignvariableop_46_dense_718_bias:7?
1assignvariableop_47_batch_normalization_647_gamma:7>
0assignvariableop_48_batch_normalization_647_beta:7E
7assignvariableop_49_batch_normalization_647_moving_mean:7I
;assignvariableop_50_batch_normalization_647_moving_variance:76
$assignvariableop_51_dense_719_kernel:770
"assignvariableop_52_dense_719_bias:7?
1assignvariableop_53_batch_normalization_648_gamma:7>
0assignvariableop_54_batch_normalization_648_beta:7E
7assignvariableop_55_batch_normalization_648_moving_mean:7I
;assignvariableop_56_batch_normalization_648_moving_variance:76
$assignvariableop_57_dense_720_kernel:70
"assignvariableop_58_dense_720_bias:'
assignvariableop_59_adam_iter:	 )
assignvariableop_60_adam_beta_1: )
assignvariableop_61_adam_beta_2: (
assignvariableop_62_adam_decay: #
assignvariableop_63_total: %
assignvariableop_64_count_1: =
+assignvariableop_65_adam_dense_711_kernel_m:i7
)assignvariableop_66_adam_dense_711_bias_m:iF
8assignvariableop_67_adam_batch_normalization_640_gamma_m:iE
7assignvariableop_68_adam_batch_normalization_640_beta_m:i=
+assignvariableop_69_adam_dense_712_kernel_m:ii7
)assignvariableop_70_adam_dense_712_bias_m:iF
8assignvariableop_71_adam_batch_normalization_641_gamma_m:iE
7assignvariableop_72_adam_batch_normalization_641_beta_m:i=
+assignvariableop_73_adam_dense_713_kernel_m:ii7
)assignvariableop_74_adam_dense_713_bias_m:iF
8assignvariableop_75_adam_batch_normalization_642_gamma_m:iE
7assignvariableop_76_adam_batch_normalization_642_beta_m:i=
+assignvariableop_77_adam_dense_714_kernel_m:ii7
)assignvariableop_78_adam_dense_714_bias_m:iF
8assignvariableop_79_adam_batch_normalization_643_gamma_m:iE
7assignvariableop_80_adam_batch_normalization_643_beta_m:i=
+assignvariableop_81_adam_dense_715_kernel_m:ii7
)assignvariableop_82_adam_dense_715_bias_m:iF
8assignvariableop_83_adam_batch_normalization_644_gamma_m:iE
7assignvariableop_84_adam_batch_normalization_644_beta_m:i=
+assignvariableop_85_adam_dense_716_kernel_m:i=7
)assignvariableop_86_adam_dense_716_bias_m:=F
8assignvariableop_87_adam_batch_normalization_645_gamma_m:=E
7assignvariableop_88_adam_batch_normalization_645_beta_m:==
+assignvariableop_89_adam_dense_717_kernel_m:==7
)assignvariableop_90_adam_dense_717_bias_m:=F
8assignvariableop_91_adam_batch_normalization_646_gamma_m:=E
7assignvariableop_92_adam_batch_normalization_646_beta_m:==
+assignvariableop_93_adam_dense_718_kernel_m:=77
)assignvariableop_94_adam_dense_718_bias_m:7F
8assignvariableop_95_adam_batch_normalization_647_gamma_m:7E
7assignvariableop_96_adam_batch_normalization_647_beta_m:7=
+assignvariableop_97_adam_dense_719_kernel_m:777
)assignvariableop_98_adam_dense_719_bias_m:7F
8assignvariableop_99_adam_batch_normalization_648_gamma_m:7F
8assignvariableop_100_adam_batch_normalization_648_beta_m:7>
,assignvariableop_101_adam_dense_720_kernel_m:78
*assignvariableop_102_adam_dense_720_bias_m:>
,assignvariableop_103_adam_dense_711_kernel_v:i8
*assignvariableop_104_adam_dense_711_bias_v:iG
9assignvariableop_105_adam_batch_normalization_640_gamma_v:iF
8assignvariableop_106_adam_batch_normalization_640_beta_v:i>
,assignvariableop_107_adam_dense_712_kernel_v:ii8
*assignvariableop_108_adam_dense_712_bias_v:iG
9assignvariableop_109_adam_batch_normalization_641_gamma_v:iF
8assignvariableop_110_adam_batch_normalization_641_beta_v:i>
,assignvariableop_111_adam_dense_713_kernel_v:ii8
*assignvariableop_112_adam_dense_713_bias_v:iG
9assignvariableop_113_adam_batch_normalization_642_gamma_v:iF
8assignvariableop_114_adam_batch_normalization_642_beta_v:i>
,assignvariableop_115_adam_dense_714_kernel_v:ii8
*assignvariableop_116_adam_dense_714_bias_v:iG
9assignvariableop_117_adam_batch_normalization_643_gamma_v:iF
8assignvariableop_118_adam_batch_normalization_643_beta_v:i>
,assignvariableop_119_adam_dense_715_kernel_v:ii8
*assignvariableop_120_adam_dense_715_bias_v:iG
9assignvariableop_121_adam_batch_normalization_644_gamma_v:iF
8assignvariableop_122_adam_batch_normalization_644_beta_v:i>
,assignvariableop_123_adam_dense_716_kernel_v:i=8
*assignvariableop_124_adam_dense_716_bias_v:=G
9assignvariableop_125_adam_batch_normalization_645_gamma_v:=F
8assignvariableop_126_adam_batch_normalization_645_beta_v:=>
,assignvariableop_127_adam_dense_717_kernel_v:==8
*assignvariableop_128_adam_dense_717_bias_v:=G
9assignvariableop_129_adam_batch_normalization_646_gamma_v:=F
8assignvariableop_130_adam_batch_normalization_646_beta_v:=>
,assignvariableop_131_adam_dense_718_kernel_v:=78
*assignvariableop_132_adam_dense_718_bias_v:7G
9assignvariableop_133_adam_batch_normalization_647_gamma_v:7F
8assignvariableop_134_adam_batch_normalization_647_beta_v:7>
,assignvariableop_135_adam_dense_719_kernel_v:778
*assignvariableop_136_adam_dense_719_bias_v:7G
9assignvariableop_137_adam_batch_normalization_648_gamma_v:7F
8assignvariableop_138_adam_batch_normalization_648_beta_v:7>
,assignvariableop_139_adam_dense_720_kernel_v:78
*assignvariableop_140_adam_dense_720_bias_v:
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
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_711_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_711_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp0assignvariableop_5_batch_normalization_640_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_640_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_640_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_8AssignVariableOp:assignvariableop_8_batch_normalization_640_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_712_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_712_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_11AssignVariableOp1assignvariableop_11_batch_normalization_641_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_641_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_641_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_14AssignVariableOp;assignvariableop_14_batch_normalization_641_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_713_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_713_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_17AssignVariableOp1assignvariableop_17_batch_normalization_642_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_642_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_19AssignVariableOp7assignvariableop_19_batch_normalization_642_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_20AssignVariableOp;assignvariableop_20_batch_normalization_642_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp$assignvariableop_21_dense_714_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_714_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_23AssignVariableOp1assignvariableop_23_batch_normalization_643_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_643_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_25AssignVariableOp7assignvariableop_25_batch_normalization_643_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_26AssignVariableOp;assignvariableop_26_batch_normalization_643_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp$assignvariableop_27_dense_715_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_715_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_29AssignVariableOp1assignvariableop_29_batch_normalization_644_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_normalization_644_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_31AssignVariableOp7assignvariableop_31_batch_normalization_644_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_32AssignVariableOp;assignvariableop_32_batch_normalization_644_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp$assignvariableop_33_dense_716_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_716_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_35AssignVariableOp1assignvariableop_35_batch_normalization_645_gammaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_645_betaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_37AssignVariableOp7assignvariableop_37_batch_normalization_645_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_38AssignVariableOp;assignvariableop_38_batch_normalization_645_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp$assignvariableop_39_dense_717_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp"assignvariableop_40_dense_717_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_41AssignVariableOp1assignvariableop_41_batch_normalization_646_gammaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_42AssignVariableOp0assignvariableop_42_batch_normalization_646_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_43AssignVariableOp7assignvariableop_43_batch_normalization_646_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_44AssignVariableOp;assignvariableop_44_batch_normalization_646_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp$assignvariableop_45_dense_718_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp"assignvariableop_46_dense_718_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_47AssignVariableOp1assignvariableop_47_batch_normalization_647_gammaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_48AssignVariableOp0assignvariableop_48_batch_normalization_647_betaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_49AssignVariableOp7assignvariableop_49_batch_normalization_647_moving_meanIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_50AssignVariableOp;assignvariableop_50_batch_normalization_647_moving_varianceIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp$assignvariableop_51_dense_719_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp"assignvariableop_52_dense_719_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_53AssignVariableOp1assignvariableop_53_batch_normalization_648_gammaIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_54AssignVariableOp0assignvariableop_54_batch_normalization_648_betaIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_55AssignVariableOp7assignvariableop_55_batch_normalization_648_moving_meanIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_56AssignVariableOp;assignvariableop_56_batch_normalization_648_moving_varianceIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp$assignvariableop_57_dense_720_kernelIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp"assignvariableop_58_dense_720_biasIdentity_58:output:0"/device:CPU:0*
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
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_dense_711_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_dense_711_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_67AssignVariableOp8assignvariableop_67_adam_batch_normalization_640_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_68AssignVariableOp7assignvariableop_68_adam_batch_normalization_640_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_dense_712_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_dense_712_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_71AssignVariableOp8assignvariableop_71_adam_batch_normalization_641_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_72AssignVariableOp7assignvariableop_72_adam_batch_normalization_641_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_dense_713_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_dense_713_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_75AssignVariableOp8assignvariableop_75_adam_batch_normalization_642_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_76AssignVariableOp7assignvariableop_76_adam_batch_normalization_642_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_dense_714_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_dense_714_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_79AssignVariableOp8assignvariableop_79_adam_batch_normalization_643_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_80AssignVariableOp7assignvariableop_80_adam_batch_normalization_643_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_dense_715_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_dense_715_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_83AssignVariableOp8assignvariableop_83_adam_batch_normalization_644_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_84AssignVariableOp7assignvariableop_84_adam_batch_normalization_644_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_dense_716_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_dense_716_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_87AssignVariableOp8assignvariableop_87_adam_batch_normalization_645_gamma_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_88AssignVariableOp7assignvariableop_88_adam_batch_normalization_645_beta_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_dense_717_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_dense_717_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_91AssignVariableOp8assignvariableop_91_adam_batch_normalization_646_gamma_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_92AssignVariableOp7assignvariableop_92_adam_batch_normalization_646_beta_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_dense_718_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_dense_718_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_95AssignVariableOp8assignvariableop_95_adam_batch_normalization_647_gamma_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_96AssignVariableOp7assignvariableop_96_adam_batch_normalization_647_beta_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_dense_719_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_dense_719_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_99AssignVariableOp8assignvariableop_99_adam_batch_normalization_648_gamma_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_100AssignVariableOp8assignvariableop_100_adam_batch_normalization_648_beta_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_dense_720_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_dense_720_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_103AssignVariableOp,assignvariableop_103_adam_dense_711_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_104AssignVariableOp*assignvariableop_104_adam_dense_711_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_105AssignVariableOp9assignvariableop_105_adam_batch_normalization_640_gamma_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_106AssignVariableOp8assignvariableop_106_adam_batch_normalization_640_beta_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_107AssignVariableOp,assignvariableop_107_adam_dense_712_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_108AssignVariableOp*assignvariableop_108_adam_dense_712_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_109AssignVariableOp9assignvariableop_109_adam_batch_normalization_641_gamma_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_110AssignVariableOp8assignvariableop_110_adam_batch_normalization_641_beta_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_111AssignVariableOp,assignvariableop_111_adam_dense_713_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_112AssignVariableOp*assignvariableop_112_adam_dense_713_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_113AssignVariableOp9assignvariableop_113_adam_batch_normalization_642_gamma_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_114AssignVariableOp8assignvariableop_114_adam_batch_normalization_642_beta_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_115AssignVariableOp,assignvariableop_115_adam_dense_714_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_116AssignVariableOp*assignvariableop_116_adam_dense_714_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_117AssignVariableOp9assignvariableop_117_adam_batch_normalization_643_gamma_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_118AssignVariableOp8assignvariableop_118_adam_batch_normalization_643_beta_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_119AssignVariableOp,assignvariableop_119_adam_dense_715_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_120AssignVariableOp*assignvariableop_120_adam_dense_715_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_121AssignVariableOp9assignvariableop_121_adam_batch_normalization_644_gamma_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_122AssignVariableOp8assignvariableop_122_adam_batch_normalization_644_beta_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_123AssignVariableOp,assignvariableop_123_adam_dense_716_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_124AssignVariableOp*assignvariableop_124_adam_dense_716_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_125AssignVariableOp9assignvariableop_125_adam_batch_normalization_645_gamma_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_126AssignVariableOp8assignvariableop_126_adam_batch_normalization_645_beta_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_127AssignVariableOp,assignvariableop_127_adam_dense_717_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_128AssignVariableOp*assignvariableop_128_adam_dense_717_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_129AssignVariableOp9assignvariableop_129_adam_batch_normalization_646_gamma_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_130AssignVariableOp8assignvariableop_130_adam_batch_normalization_646_beta_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_131AssignVariableOp,assignvariableop_131_adam_dense_718_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_132AssignVariableOp*assignvariableop_132_adam_dense_718_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_133AssignVariableOp9assignvariableop_133_adam_batch_normalization_647_gamma_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_134AssignVariableOp8assignvariableop_134_adam_batch_normalization_647_beta_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_135AssignVariableOp,assignvariableop_135_adam_dense_719_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_136AssignVariableOp*assignvariableop_136_adam_dense_719_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_137AssignVariableOp9assignvariableop_137_adam_batch_normalization_648_gamma_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_138AssignVariableOp8assignvariableop_138_adam_batch_normalization_648_beta_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_139AssignVariableOp,assignvariableop_139_adam_dense_720_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_140AssignVariableOp*assignvariableop_140_adam_dense_720_bias_vIdentity_140:output:0"/device:CPU:0*
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
¥
Þ
F__inference_dense_718_layer_call_and_return_conditional_losses_1125999

inputs0
matmul_readvariableop_resource:=7-
biasadd_readvariableop_resource:7
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_718/kernel/Regularizer/Abs/ReadVariableOp¢2dense_718/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:=7*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:7*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7g
"dense_718/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/dense_718/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:=7*
dtype0
 dense_718/kernel/Regularizer/AbsAbs7dense_718/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_718/kernel/Regularizer/SumSum$dense_718/kernel/Regularizer/Abs:y:0-dense_718/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_718/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ù= 
 dense_718/kernel/Regularizer/mulMul+dense_718/kernel/Regularizer/mul/x:output:0)dense_718/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_718/kernel/Regularizer/addAddV2+dense_718/kernel/Regularizer/Const:output:0$dense_718/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2dense_718/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:=7*
dtype0
#dense_718/kernel/Regularizer/SquareSquare:dense_718/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:=7u
$dense_718/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_718/kernel/Regularizer/Sum_1Sum'dense_718/kernel/Regularizer/Square:y:0-dense_718/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_718/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *-=¦
"dense_718/kernel/Regularizer/mul_1Mul-dense_718/kernel/Regularizer/mul_1/x:output:0+dense_718/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_718/kernel/Regularizer/add_1AddV2$dense_718/kernel/Regularizer/add:z:0&dense_718/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7Þ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_718/kernel/Regularizer/Abs/ReadVariableOp3^dense_718/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ=: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_718/kernel/Regularizer/Abs/ReadVariableOp/dense_718/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_718/kernel/Regularizer/Square/ReadVariableOp2dense_718/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_640_layer_call_fn_1125039

inputs
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_640_layer_call_and_return_conditional_losses_1120785o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_643_layer_call_and_return_conditional_losses_1121078

inputs5
'assignmovingavg_readvariableop_resource:i7
)assignmovingavg_1_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i/
!batchnorm_readvariableop_resource:i
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:i
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿil
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:i*
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
:i*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ix
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i¬
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
:i*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:i~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i´
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿih
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:iv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_643_layer_call_and_return_conditional_losses_1125523

inputs5
'assignmovingavg_readvariableop_resource:i7
)assignmovingavg_1_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i/
!batchnorm_readvariableop_resource:i
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:i
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿil
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:i*
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
:i*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ix
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i¬
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
:i*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:i~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i´
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿih
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:iv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
÷
ã
/__inference_sequential_71_layer_call_fn_1123010
normalization_71_input
unknown
	unknown_0
	unknown_1:i
	unknown_2:i
	unknown_3:i
	unknown_4:i
	unknown_5:i
	unknown_6:i
	unknown_7:ii
	unknown_8:i
	unknown_9:i

unknown_10:i

unknown_11:i

unknown_12:i

unknown_13:ii

unknown_14:i

unknown_15:i

unknown_16:i

unknown_17:i

unknown_18:i

unknown_19:ii

unknown_20:i

unknown_21:i

unknown_22:i

unknown_23:i

unknown_24:i

unknown_25:ii

unknown_26:i

unknown_27:i

unknown_28:i

unknown_29:i

unknown_30:i

unknown_31:i=

unknown_32:=

unknown_33:=

unknown_34:=

unknown_35:=

unknown_36:=

unknown_37:==

unknown_38:=

unknown_39:=

unknown_40:=

unknown_41:=

unknown_42:=

unknown_43:=7

unknown_44:7

unknown_45:7

unknown_46:7

unknown_47:7

unknown_48:7

unknown_49:77

unknown_50:7

unknown_51:7

unknown_52:7

unknown_53:7

unknown_54:7

unknown_55:7

unknown_56:
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallnormalization_71_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8 *S
fNRL
J__inference_sequential_71_layer_call_and_return_conditional_losses_1122770o
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
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_71_input:$ 

_output_shapes

::$ 

_output_shapes

:
Æ

+__inference_dense_720_layer_call_fn_1126237

inputs
unknown:7
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
F__inference_dense_720_layer_call_and_return_conditional_losses_1121946o
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
:ÿÿÿÿÿÿÿÿÿ7: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_641_layer_call_and_return_conditional_losses_1125255

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿi:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_642_layer_call_and_return_conditional_losses_1125384

inputs5
'assignmovingavg_readvariableop_resource:i7
)assignmovingavg_1_readvariableop_resource:i3
%batchnorm_mul_readvariableop_resource:i/
!batchnorm_readvariableop_resource:i
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:i
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿil
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:i*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:i*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:i*
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
:i*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ix
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:i¬
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
:i*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:i~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:i´
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
:iP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:i~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:i*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ic
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿih
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:iv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:i*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ir
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿib
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿiê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
_user_specified_nameinputs

ã
__inference_loss_fn_5_1126367J
8dense_716_kernel_regularizer_abs_readvariableop_resource:i=
identity¢/dense_716/kernel/Regularizer/Abs/ReadVariableOp¢2dense_716/kernel/Regularizer/Square/ReadVariableOpg
"dense_716/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
/dense_716/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8dense_716_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:i=*
dtype0
 dense_716/kernel/Regularizer/AbsAbs7dense_716/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
 dense_716/kernel/Regularizer/SumSum$dense_716/kernel/Regularizer/Abs:y:0-dense_716/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"dense_716/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤= 
 dense_716/kernel/Regularizer/mulMul+dense_716/kernel/Regularizer/mul/x:output:0)dense_716/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 dense_716/kernel/Regularizer/addAddV2+dense_716/kernel/Regularizer/Const:output:0$dense_716/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: «
2dense_716/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8dense_716_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:i=*
dtype0
#dense_716/kernel/Regularizer/SquareSquare:dense_716/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:i=u
$dense_716/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ¢
"dense_716/kernel/Regularizer/Sum_1Sum'dense_716/kernel/Regularizer/Square:y:0-dense_716/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$dense_716/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *5£=¦
"dense_716/kernel/Regularizer/mul_1Mul-dense_716/kernel/Regularizer/mul_1/x:output:0+dense_716/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"dense_716/kernel/Regularizer/add_1AddV2$dense_716/kernel/Regularizer/add:z:0&dense_716/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
IdentityIdentity&dense_716/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ­
NoOpNoOp0^dense_716/kernel/Regularizer/Abs/ReadVariableOp3^dense_716/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_716/kernel/Regularizer/Abs/ReadVariableOp/dense_716/kernel/Regularizer/Abs/ReadVariableOp2h
2dense_716/kernel/Regularizer/Square/ReadVariableOp2dense_716/kernel/Regularizer/Square/ReadVariableOp
®
Ô
9__inference_batch_normalization_644_layer_call_fn_1125595

inputs
unknown:i
	unknown_0:i
	unknown_1:i
	unknown_2:i
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_644_layer_call_and_return_conditional_losses_1121113o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿi: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
 
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
normalization_71_input?
(serving_default_normalization_71_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_7200
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Î
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

decay0m½1m¾9m¿:mÀImÁJmÂRmÃSmÄbmÅcmÆkmÇlmÈ{mÉ|mÊ	mË	mÌ	mÍ	mÎ	mÏ	mÐ	­mÑ	®mÒ	¶mÓ	·mÔ	ÆmÕ	ÇmÖ	Ïm×	ÐmØ	ßmÙ	àmÚ	èmÛ	émÜ	ømÝ	ùmÞ	mß	mà	má	mâ0vã1vä9vå:væIvçJvèRvéSvêbvëcvìkvílvî{vï|vð	vñ	vò	vó	vô	võ	vö	­v÷	®vø	¶vù	·vú	Ævû	Çvü	Ïvý	Ðvþ	ßvÿ	àv	èv	év	øv	ùv	v	v	v	v"
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
h
0
1
2
 3
¡4
¢5
£6
¤7
¥8"
trackable_list_wrapper
Ï
¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
%_default_save_signature
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
2
/__inference_sequential_71_layer_call_fn_1122207
/__inference_sequential_71_layer_call_fn_1123842
/__inference_sequential_71_layer_call_fn_1123963
/__inference_sequential_71_layer_call_fn_1123010À
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
J__inference_sequential_71_layer_call_and_return_conditional_losses_1124322
J__inference_sequential_71_layer_call_and_return_conditional_losses_1124807
J__inference_sequential_71_layer_call_and_return_conditional_losses_1123296
J__inference_sequential_71_layer_call_and_return_conditional_losses_1123582À
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
"__inference__wrapped_model_1120761normalization_71_input"
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
«serving_default"
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
À2½
__inference_adapt_step_1124977
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
": i2dense_711/kernel
:i2dense_711/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
(
0"
trackable_list_wrapper
²
¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_711_layer_call_fn_1125001¢
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
F__inference_dense_711_layer_call_and_return_conditional_losses_1125026¢
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
+:)i2batch_normalization_640/gamma
*:(i2batch_normalization_640/beta
3:1i (2#batch_normalization_640/moving_mean
7:5i (2'batch_normalization_640/moving_variance
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
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_640_layer_call_fn_1125039
9__inference_batch_normalization_640_layer_call_fn_1125052´
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
T__inference_batch_normalization_640_layer_call_and_return_conditional_losses_1125072
T__inference_batch_normalization_640_layer_call_and_return_conditional_losses_1125106´
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
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_640_layer_call_fn_1125111¢
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
L__inference_leaky_re_lu_640_layer_call_and_return_conditional_losses_1125116¢
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
": ii2dense_712/kernel
:i2dense_712/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
²
»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_712_layer_call_fn_1125140¢
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
F__inference_dense_712_layer_call_and_return_conditional_losses_1125165¢
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
+:)i2batch_normalization_641/gamma
*:(i2batch_normalization_641/beta
3:1i (2#batch_normalization_641/moving_mean
7:5i (2'batch_normalization_641/moving_variance
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
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_641_layer_call_fn_1125178
9__inference_batch_normalization_641_layer_call_fn_1125191´
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
T__inference_batch_normalization_641_layer_call_and_return_conditional_losses_1125211
T__inference_batch_normalization_641_layer_call_and_return_conditional_losses_1125245´
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
Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_641_layer_call_fn_1125250¢
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
L__inference_leaky_re_lu_641_layer_call_and_return_conditional_losses_1125255¢
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
": ii2dense_713/kernel
:i2dense_713/bias
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
(
0"
trackable_list_wrapper
²
Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_713_layer_call_fn_1125279¢
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
F__inference_dense_713_layer_call_and_return_conditional_losses_1125304¢
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
+:)i2batch_normalization_642/gamma
*:(i2batch_normalization_642/beta
3:1i (2#batch_normalization_642/moving_mean
7:5i (2'batch_normalization_642/moving_variance
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
Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_642_layer_call_fn_1125317
9__inference_batch_normalization_642_layer_call_fn_1125330´
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
T__inference_batch_normalization_642_layer_call_and_return_conditional_losses_1125350
T__inference_batch_normalization_642_layer_call_and_return_conditional_losses_1125384´
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
Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_642_layer_call_fn_1125389¢
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
L__inference_leaky_re_lu_642_layer_call_and_return_conditional_losses_1125394¢
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
": ii2dense_714/kernel
:i2dense_714/bias
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
(
 0"
trackable_list_wrapper
µ
Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_714_layer_call_fn_1125418¢
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
F__inference_dense_714_layer_call_and_return_conditional_losses_1125443¢
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
+:)i2batch_normalization_643/gamma
*:(i2batch_normalization_643/beta
3:1i (2#batch_normalization_643/moving_mean
7:5i (2'batch_normalization_643/moving_variance
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
Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_643_layer_call_fn_1125456
9__inference_batch_normalization_643_layer_call_fn_1125469´
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
T__inference_batch_normalization_643_layer_call_and_return_conditional_losses_1125489
T__inference_batch_normalization_643_layer_call_and_return_conditional_losses_1125523´
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
ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_643_layer_call_fn_1125528¢
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
L__inference_leaky_re_lu_643_layer_call_and_return_conditional_losses_1125533¢
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
": ii2dense_715/kernel
:i2dense_715/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
(
¡0"
trackable_list_wrapper
¸
ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_715_layer_call_fn_1125557¢
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
F__inference_dense_715_layer_call_and_return_conditional_losses_1125582¢
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
+:)i2batch_normalization_644/gamma
*:(i2batch_normalization_644/beta
3:1i (2#batch_normalization_644/moving_mean
7:5i (2'batch_normalization_644/moving_variance
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
ínon_trainable_variables
îlayers
ïmetrics
 ðlayer_regularization_losses
ñlayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_644_layer_call_fn_1125595
9__inference_batch_normalization_644_layer_call_fn_1125608´
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
T__inference_batch_normalization_644_layer_call_and_return_conditional_losses_1125628
T__inference_batch_normalization_644_layer_call_and_return_conditional_losses_1125662´
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
ònon_trainable_variables
ólayers
ômetrics
 õlayer_regularization_losses
ölayer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_644_layer_call_fn_1125667¢
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
L__inference_leaky_re_lu_644_layer_call_and_return_conditional_losses_1125672¢
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
": i=2dense_716/kernel
:=2dense_716/bias
0
­0
®1"
trackable_list_wrapper
0
­0
®1"
trackable_list_wrapper
(
¢0"
trackable_list_wrapper
¸
÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
¯	variables
°trainable_variables
±regularization_losses
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_716_layer_call_fn_1125696¢
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
F__inference_dense_716_layer_call_and_return_conditional_losses_1125721¢
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
+:)=2batch_normalization_645/gamma
*:(=2batch_normalization_645/beta
3:1= (2#batch_normalization_645/moving_mean
7:5= (2'batch_normalization_645/moving_variance
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
ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
º	variables
»trainable_variables
¼regularization_losses
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_645_layer_call_fn_1125734
9__inference_batch_normalization_645_layer_call_fn_1125747´
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
T__inference_batch_normalization_645_layer_call_and_return_conditional_losses_1125767
T__inference_batch_normalization_645_layer_call_and_return_conditional_losses_1125801´
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
À	variables
Átrainable_variables
Âregularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_645_layer_call_fn_1125806¢
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
L__inference_leaky_re_lu_645_layer_call_and_return_conditional_losses_1125811¢
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
": ==2dense_717/kernel
:=2dense_717/bias
0
Æ0
Ç1"
trackable_list_wrapper
0
Æ0
Ç1"
trackable_list_wrapper
(
£0"
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
È	variables
Étrainable_variables
Êregularization_losses
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_717_layer_call_fn_1125835¢
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
F__inference_dense_717_layer_call_and_return_conditional_losses_1125860¢
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
+:)=2batch_normalization_646/gamma
*:(=2batch_normalization_646/beta
3:1= (2#batch_normalization_646/moving_mean
7:5= (2'batch_normalization_646/moving_variance
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ó	variables
Ôtrainable_variables
Õregularization_losses
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_646_layer_call_fn_1125873
9__inference_batch_normalization_646_layer_call_fn_1125886´
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
T__inference_batch_normalization_646_layer_call_and_return_conditional_losses_1125906
T__inference_batch_normalization_646_layer_call_and_return_conditional_losses_1125940´
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ù	variables
Útrainable_variables
Ûregularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_646_layer_call_fn_1125945¢
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
L__inference_leaky_re_lu_646_layer_call_and_return_conditional_losses_1125950¢
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
": =72dense_718/kernel
:72dense_718/bias
0
ß0
à1"
trackable_list_wrapper
0
ß0
à1"
trackable_list_wrapper
(
¤0"
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
á	variables
âtrainable_variables
ãregularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_718_layer_call_fn_1125974¢
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
F__inference_dense_718_layer_call_and_return_conditional_losses_1125999¢
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
+:)72batch_normalization_647/gamma
*:(72batch_normalization_647/beta
3:17 (2#batch_normalization_647/moving_mean
7:57 (2'batch_normalization_647/moving_variance
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ì	variables
ítrainable_variables
îregularization_losses
ð__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_647_layer_call_fn_1126012
9__inference_batch_normalization_647_layer_call_fn_1126025´
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
T__inference_batch_normalization_647_layer_call_and_return_conditional_losses_1126045
T__inference_batch_normalization_647_layer_call_and_return_conditional_losses_1126079´
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
non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
ò	variables
ótrainable_variables
ôregularization_losses
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_647_layer_call_fn_1126084¢
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
L__inference_leaky_re_lu_647_layer_call_and_return_conditional_losses_1126089¢
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
": 772dense_719/kernel
:72dense_719/bias
0
ø0
ù1"
trackable_list_wrapper
0
ø0
ù1"
trackable_list_wrapper
(
¥0"
trackable_list_wrapper
¸
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
ú	variables
ûtrainable_variables
üregularization_losses
þ__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_719_layer_call_fn_1126113¢
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
F__inference_dense_719_layer_call_and_return_conditional_losses_1126138¢
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
+:)72batch_normalization_648/gamma
*:(72batch_normalization_648/beta
3:17 (2#batch_normalization_648/moving_mean
7:57 (2'batch_normalization_648/moving_variance
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
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_648_layer_call_fn_1126151
9__inference_batch_normalization_648_layer_call_fn_1126164´
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
T__inference_batch_normalization_648_layer_call_and_return_conditional_losses_1126184
T__inference_batch_normalization_648_layer_call_and_return_conditional_losses_1126218´
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
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_648_layer_call_fn_1126223¢
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
L__inference_leaky_re_lu_648_layer_call_and_return_conditional_losses_1126228¢
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
": 72dense_720/kernel
:2dense_720/bias
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
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_720_layer_call_fn_1126237¢
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
F__inference_dense_720_layer_call_and_return_conditional_losses_1126247¢
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
´2±
__inference_loss_fn_0_1126267
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
´2±
__inference_loss_fn_1_1126287
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
´2±
__inference_loss_fn_2_1126307
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
´2±
__inference_loss_fn_3_1126327
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
´2±
__inference_loss_fn_4_1126347
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
´2±
__inference_loss_fn_5_1126367
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
´2±
__inference_loss_fn_6_1126387
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
´2±
__inference_loss_fn_7_1126407
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
´2±
__inference_loss_fn_8_1126427
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
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
¸0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
%__inference_signature_wrapper_1124930normalization_71_input"
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
(
0"
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
(
0"
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
(
0"
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
(
 0"
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
(
¡0"
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
(
¢0"
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
(
£0"
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
(
¤0"
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
(
¥0"
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

¹total

ºcount
»	variables
¼	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
¹0
º1"
trackable_list_wrapper
.
»	variables"
_generic_user_object
':%i2Adam/dense_711/kernel/m
!:i2Adam/dense_711/bias/m
0:.i2$Adam/batch_normalization_640/gamma/m
/:-i2#Adam/batch_normalization_640/beta/m
':%ii2Adam/dense_712/kernel/m
!:i2Adam/dense_712/bias/m
0:.i2$Adam/batch_normalization_641/gamma/m
/:-i2#Adam/batch_normalization_641/beta/m
':%ii2Adam/dense_713/kernel/m
!:i2Adam/dense_713/bias/m
0:.i2$Adam/batch_normalization_642/gamma/m
/:-i2#Adam/batch_normalization_642/beta/m
':%ii2Adam/dense_714/kernel/m
!:i2Adam/dense_714/bias/m
0:.i2$Adam/batch_normalization_643/gamma/m
/:-i2#Adam/batch_normalization_643/beta/m
':%ii2Adam/dense_715/kernel/m
!:i2Adam/dense_715/bias/m
0:.i2$Adam/batch_normalization_644/gamma/m
/:-i2#Adam/batch_normalization_644/beta/m
':%i=2Adam/dense_716/kernel/m
!:=2Adam/dense_716/bias/m
0:.=2$Adam/batch_normalization_645/gamma/m
/:-=2#Adam/batch_normalization_645/beta/m
':%==2Adam/dense_717/kernel/m
!:=2Adam/dense_717/bias/m
0:.=2$Adam/batch_normalization_646/gamma/m
/:-=2#Adam/batch_normalization_646/beta/m
':%=72Adam/dense_718/kernel/m
!:72Adam/dense_718/bias/m
0:.72$Adam/batch_normalization_647/gamma/m
/:-72#Adam/batch_normalization_647/beta/m
':%772Adam/dense_719/kernel/m
!:72Adam/dense_719/bias/m
0:.72$Adam/batch_normalization_648/gamma/m
/:-72#Adam/batch_normalization_648/beta/m
':%72Adam/dense_720/kernel/m
!:2Adam/dense_720/bias/m
':%i2Adam/dense_711/kernel/v
!:i2Adam/dense_711/bias/v
0:.i2$Adam/batch_normalization_640/gamma/v
/:-i2#Adam/batch_normalization_640/beta/v
':%ii2Adam/dense_712/kernel/v
!:i2Adam/dense_712/bias/v
0:.i2$Adam/batch_normalization_641/gamma/v
/:-i2#Adam/batch_normalization_641/beta/v
':%ii2Adam/dense_713/kernel/v
!:i2Adam/dense_713/bias/v
0:.i2$Adam/batch_normalization_642/gamma/v
/:-i2#Adam/batch_normalization_642/beta/v
':%ii2Adam/dense_714/kernel/v
!:i2Adam/dense_714/bias/v
0:.i2$Adam/batch_normalization_643/gamma/v
/:-i2#Adam/batch_normalization_643/beta/v
':%ii2Adam/dense_715/kernel/v
!:i2Adam/dense_715/bias/v
0:.i2$Adam/batch_normalization_644/gamma/v
/:-i2#Adam/batch_normalization_644/beta/v
':%i=2Adam/dense_716/kernel/v
!:=2Adam/dense_716/bias/v
0:.=2$Adam/batch_normalization_645/gamma/v
/:-=2#Adam/batch_normalization_645/beta/v
':%==2Adam/dense_717/kernel/v
!:=2Adam/dense_717/bias/v
0:.=2$Adam/batch_normalization_646/gamma/v
/:-=2#Adam/batch_normalization_646/beta/v
':%=72Adam/dense_718/kernel/v
!:72Adam/dense_718/bias/v
0:.72$Adam/batch_normalization_647/gamma/v
/:-72#Adam/batch_normalization_647/beta/v
':%772Adam/dense_719/kernel/v
!:72Adam/dense_719/bias/v
0:.72$Adam/batch_normalization_648/gamma/v
/:-72#Adam/batch_normalization_648/beta/v
':%72Adam/dense_720/kernel/v
!:2Adam/dense_720/bias/v
	J
Const
J	
Const_1
"__inference__wrapped_model_1120761Ú`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøù?¢<
5¢2
0-
normalization_71_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_720# 
	dense_720ÿÿÿÿÿÿÿÿÿg
__inference_adapt_step_1124977E-+,:¢7
0¢-
+(¢
 	IteratorSpec 
ª "
 º
T__inference_batch_normalization_640_layer_call_and_return_conditional_losses_1125072b<9;:3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 º
T__inference_batch_normalization_640_layer_call_and_return_conditional_losses_1125106b;<9:3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 
9__inference_batch_normalization_640_layer_call_fn_1125039U<9;:3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p 
ª "ÿÿÿÿÿÿÿÿÿi
9__inference_batch_normalization_640_layer_call_fn_1125052U;<9:3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p
ª "ÿÿÿÿÿÿÿÿÿiº
T__inference_batch_normalization_641_layer_call_and_return_conditional_losses_1125211bURTS3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 º
T__inference_batch_normalization_641_layer_call_and_return_conditional_losses_1125245bTURS3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 
9__inference_batch_normalization_641_layer_call_fn_1125178UURTS3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p 
ª "ÿÿÿÿÿÿÿÿÿi
9__inference_batch_normalization_641_layer_call_fn_1125191UTURS3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p
ª "ÿÿÿÿÿÿÿÿÿiº
T__inference_batch_normalization_642_layer_call_and_return_conditional_losses_1125350bnkml3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 º
T__inference_batch_normalization_642_layer_call_and_return_conditional_losses_1125384bmnkl3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 
9__inference_batch_normalization_642_layer_call_fn_1125317Unkml3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p 
ª "ÿÿÿÿÿÿÿÿÿi
9__inference_batch_normalization_642_layer_call_fn_1125330Umnkl3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p
ª "ÿÿÿÿÿÿÿÿÿi¾
T__inference_batch_normalization_643_layer_call_and_return_conditional_losses_1125489f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 ¾
T__inference_batch_normalization_643_layer_call_and_return_conditional_losses_1125523f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 
9__inference_batch_normalization_643_layer_call_fn_1125456Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p 
ª "ÿÿÿÿÿÿÿÿÿi
9__inference_batch_normalization_643_layer_call_fn_1125469Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p
ª "ÿÿÿÿÿÿÿÿÿi¾
T__inference_batch_normalization_644_layer_call_and_return_conditional_losses_1125628f 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 ¾
T__inference_batch_normalization_644_layer_call_and_return_conditional_losses_1125662f 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 
9__inference_batch_normalization_644_layer_call_fn_1125595Y 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p 
ª "ÿÿÿÿÿÿÿÿÿi
9__inference_batch_normalization_644_layer_call_fn_1125608Y 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿi
p
ª "ÿÿÿÿÿÿÿÿÿi¾
T__inference_batch_normalization_645_layer_call_and_return_conditional_losses_1125767f¹¶¸·3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 ¾
T__inference_batch_normalization_645_layer_call_and_return_conditional_losses_1125801f¸¹¶·3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 
9__inference_batch_normalization_645_layer_call_fn_1125734Y¹¶¸·3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p 
ª "ÿÿÿÿÿÿÿÿÿ=
9__inference_batch_normalization_645_layer_call_fn_1125747Y¸¹¶·3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p
ª "ÿÿÿÿÿÿÿÿÿ=¾
T__inference_batch_normalization_646_layer_call_and_return_conditional_losses_1125906fÒÏÑÐ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 ¾
T__inference_batch_normalization_646_layer_call_and_return_conditional_losses_1125940fÑÒÏÐ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 
9__inference_batch_normalization_646_layer_call_fn_1125873YÒÏÑÐ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p 
ª "ÿÿÿÿÿÿÿÿÿ=
9__inference_batch_normalization_646_layer_call_fn_1125886YÑÒÏÐ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ=
p
ª "ÿÿÿÿÿÿÿÿÿ=¾
T__inference_batch_normalization_647_layer_call_and_return_conditional_losses_1126045fëèêé3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 ¾
T__inference_batch_normalization_647_layer_call_and_return_conditional_losses_1126079fêëèé3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 
9__inference_batch_normalization_647_layer_call_fn_1126012Yëèêé3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p 
ª "ÿÿÿÿÿÿÿÿÿ7
9__inference_batch_normalization_647_layer_call_fn_1126025Yêëèé3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p
ª "ÿÿÿÿÿÿÿÿÿ7¾
T__inference_batch_normalization_648_layer_call_and_return_conditional_losses_1126184f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 ¾
T__inference_batch_normalization_648_layer_call_and_return_conditional_losses_1126218f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 
9__inference_batch_normalization_648_layer_call_fn_1126151Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p 
ª "ÿÿÿÿÿÿÿÿÿ7
9__inference_batch_normalization_648_layer_call_fn_1126164Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p
ª "ÿÿÿÿÿÿÿÿÿ7¦
F__inference_dense_711_layer_call_and_return_conditional_losses_1125026\01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 ~
+__inference_dense_711_layer_call_fn_1125001O01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿi¦
F__inference_dense_712_layer_call_and_return_conditional_losses_1125165\IJ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 ~
+__inference_dense_712_layer_call_fn_1125140OIJ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "ÿÿÿÿÿÿÿÿÿi¦
F__inference_dense_713_layer_call_and_return_conditional_losses_1125304\bc/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 ~
+__inference_dense_713_layer_call_fn_1125279Obc/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "ÿÿÿÿÿÿÿÿÿi¦
F__inference_dense_714_layer_call_and_return_conditional_losses_1125443\{|/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 ~
+__inference_dense_714_layer_call_fn_1125418O{|/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "ÿÿÿÿÿÿÿÿÿi¨
F__inference_dense_715_layer_call_and_return_conditional_losses_1125582^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 
+__inference_dense_715_layer_call_fn_1125557Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "ÿÿÿÿÿÿÿÿÿi¨
F__inference_dense_716_layer_call_and_return_conditional_losses_1125721^­®/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 
+__inference_dense_716_layer_call_fn_1125696Q­®/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "ÿÿÿÿÿÿÿÿÿ=¨
F__inference_dense_717_layer_call_and_return_conditional_losses_1125860^ÆÇ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 
+__inference_dense_717_layer_call_fn_1125835QÆÇ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "ÿÿÿÿÿÿÿÿÿ=¨
F__inference_dense_718_layer_call_and_return_conditional_losses_1125999^ßà/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 
+__inference_dense_718_layer_call_fn_1125974Qßà/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "ÿÿÿÿÿÿÿÿÿ7¨
F__inference_dense_719_layer_call_and_return_conditional_losses_1126138^øù/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 
+__inference_dense_719_layer_call_fn_1126113Qøù/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "ÿÿÿÿÿÿÿÿÿ7¨
F__inference_dense_720_layer_call_and_return_conditional_losses_1126247^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_720_layer_call_fn_1126237Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "ÿÿÿÿÿÿÿÿÿ¨
L__inference_leaky_re_lu_640_layer_call_and_return_conditional_losses_1125116X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 
1__inference_leaky_re_lu_640_layer_call_fn_1125111K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "ÿÿÿÿÿÿÿÿÿi¨
L__inference_leaky_re_lu_641_layer_call_and_return_conditional_losses_1125255X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 
1__inference_leaky_re_lu_641_layer_call_fn_1125250K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "ÿÿÿÿÿÿÿÿÿi¨
L__inference_leaky_re_lu_642_layer_call_and_return_conditional_losses_1125394X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 
1__inference_leaky_re_lu_642_layer_call_fn_1125389K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "ÿÿÿÿÿÿÿÿÿi¨
L__inference_leaky_re_lu_643_layer_call_and_return_conditional_losses_1125533X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 
1__inference_leaky_re_lu_643_layer_call_fn_1125528K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "ÿÿÿÿÿÿÿÿÿi¨
L__inference_leaky_re_lu_644_layer_call_and_return_conditional_losses_1125672X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "%¢"

0ÿÿÿÿÿÿÿÿÿi
 
1__inference_leaky_re_lu_644_layer_call_fn_1125667K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿi
ª "ÿÿÿÿÿÿÿÿÿi¨
L__inference_leaky_re_lu_645_layer_call_and_return_conditional_losses_1125811X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 
1__inference_leaky_re_lu_645_layer_call_fn_1125806K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "ÿÿÿÿÿÿÿÿÿ=¨
L__inference_leaky_re_lu_646_layer_call_and_return_conditional_losses_1125950X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ=
 
1__inference_leaky_re_lu_646_layer_call_fn_1125945K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ=
ª "ÿÿÿÿÿÿÿÿÿ=¨
L__inference_leaky_re_lu_647_layer_call_and_return_conditional_losses_1126089X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 
1__inference_leaky_re_lu_647_layer_call_fn_1126084K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "ÿÿÿÿÿÿÿÿÿ7¨
L__inference_leaky_re_lu_648_layer_call_and_return_conditional_losses_1126228X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 
1__inference_leaky_re_lu_648_layer_call_fn_1126223K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "ÿÿÿÿÿÿÿÿÿ7<
__inference_loss_fn_0_11262670¢

¢ 
ª " <
__inference_loss_fn_1_1126287I¢

¢ 
ª " <
__inference_loss_fn_2_1126307b¢

¢ 
ª " <
__inference_loss_fn_3_1126327{¢

¢ 
ª " =
__inference_loss_fn_4_1126347¢

¢ 
ª " =
__inference_loss_fn_5_1126367­¢

¢ 
ª " =
__inference_loss_fn_6_1126387Æ¢

¢ 
ª " =
__inference_loss_fn_7_1126407ß¢

¢ 
ª " =
__inference_loss_fn_8_1126427ø¢

¢ 
ª " ¡
J__inference_sequential_71_layer_call_and_return_conditional_losses_1123296Ò`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøùG¢D
=¢:
0-
normalization_71_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¡
J__inference_sequential_71_layer_call_and_return_conditional_losses_1123582Ò`01;<9:IJTURSbcmnkl{| ­®¸¹¶·ÆÇÑÒÏÐßàêëèéøùG¢D
=¢:
0-
normalization_71_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
J__inference_sequential_71_layer_call_and_return_conditional_losses_1124322Â`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøù7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
J__inference_sequential_71_layer_call_and_return_conditional_losses_1124807Â`01;<9:IJTURSbcmnkl{| ­®¸¹¶·ÆÇÑÒÏÐßàêëèéøù7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ù
/__inference_sequential_71_layer_call_fn_1122207Å`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøùG¢D
=¢:
0-
normalization_71_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿù
/__inference_sequential_71_layer_call_fn_1123010Å`01;<9:IJTURSbcmnkl{| ­®¸¹¶·ÆÇÑÒÏÐßàêëèéøùG¢D
=¢:
0-
normalization_71_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿé
/__inference_sequential_71_layer_call_fn_1123842µ`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøù7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿé
/__inference_sequential_71_layer_call_fn_1123963µ`01;<9:IJTURSbcmnkl{| ­®¸¹¶·ÆÇÑÒÏÐßàêëèéøù7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_signature_wrapper_1124930ô`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøùY¢V
¢ 
OªL
J
normalization_71_input0-
normalization_71_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_720# 
	dense_720ÿÿÿÿÿÿÿÿÿ