Ý·,
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68­«(
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
dense_734/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:S*!
shared_namedense_734/kernel
u
$dense_734/kernel/Read/ReadVariableOpReadVariableOpdense_734/kernel*
_output_shapes

:S*
dtype0
t
dense_734/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*
shared_namedense_734/bias
m
"dense_734/bias/Read/ReadVariableOpReadVariableOpdense_734/bias*
_output_shapes
:S*
dtype0

batch_normalization_665/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*.
shared_namebatch_normalization_665/gamma

1batch_normalization_665/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_665/gamma*
_output_shapes
:S*
dtype0

batch_normalization_665/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*-
shared_namebatch_normalization_665/beta

0batch_normalization_665/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_665/beta*
_output_shapes
:S*
dtype0

#batch_normalization_665/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*4
shared_name%#batch_normalization_665/moving_mean

7batch_normalization_665/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_665/moving_mean*
_output_shapes
:S*
dtype0
¦
'batch_normalization_665/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*8
shared_name)'batch_normalization_665/moving_variance

;batch_normalization_665/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_665/moving_variance*
_output_shapes
:S*
dtype0
|
dense_735/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:SS*!
shared_namedense_735/kernel
u
$dense_735/kernel/Read/ReadVariableOpReadVariableOpdense_735/kernel*
_output_shapes

:SS*
dtype0
t
dense_735/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*
shared_namedense_735/bias
m
"dense_735/bias/Read/ReadVariableOpReadVariableOpdense_735/bias*
_output_shapes
:S*
dtype0

batch_normalization_666/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*.
shared_namebatch_normalization_666/gamma

1batch_normalization_666/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_666/gamma*
_output_shapes
:S*
dtype0

batch_normalization_666/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*-
shared_namebatch_normalization_666/beta

0batch_normalization_666/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_666/beta*
_output_shapes
:S*
dtype0

#batch_normalization_666/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*4
shared_name%#batch_normalization_666/moving_mean

7batch_normalization_666/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_666/moving_mean*
_output_shapes
:S*
dtype0
¦
'batch_normalization_666/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*8
shared_name)'batch_normalization_666/moving_variance

;batch_normalization_666/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_666/moving_variance*
_output_shapes
:S*
dtype0
|
dense_736/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Sa*!
shared_namedense_736/kernel
u
$dense_736/kernel/Read/ReadVariableOpReadVariableOpdense_736/kernel*
_output_shapes

:Sa*
dtype0
t
dense_736/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*
shared_namedense_736/bias
m
"dense_736/bias/Read/ReadVariableOpReadVariableOpdense_736/bias*
_output_shapes
:a*
dtype0

batch_normalization_667/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*.
shared_namebatch_normalization_667/gamma

1batch_normalization_667/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_667/gamma*
_output_shapes
:a*
dtype0

batch_normalization_667/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*-
shared_namebatch_normalization_667/beta

0batch_normalization_667/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_667/beta*
_output_shapes
:a*
dtype0

#batch_normalization_667/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#batch_normalization_667/moving_mean

7batch_normalization_667/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_667/moving_mean*
_output_shapes
:a*
dtype0
¦
'batch_normalization_667/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*8
shared_name)'batch_normalization_667/moving_variance

;batch_normalization_667/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_667/moving_variance*
_output_shapes
:a*
dtype0
|
dense_737/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:aa*!
shared_namedense_737/kernel
u
$dense_737/kernel/Read/ReadVariableOpReadVariableOpdense_737/kernel*
_output_shapes

:aa*
dtype0
t
dense_737/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*
shared_namedense_737/bias
m
"dense_737/bias/Read/ReadVariableOpReadVariableOpdense_737/bias*
_output_shapes
:a*
dtype0

batch_normalization_668/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*.
shared_namebatch_normalization_668/gamma

1batch_normalization_668/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_668/gamma*
_output_shapes
:a*
dtype0

batch_normalization_668/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*-
shared_namebatch_normalization_668/beta

0batch_normalization_668/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_668/beta*
_output_shapes
:a*
dtype0

#batch_normalization_668/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#batch_normalization_668/moving_mean

7batch_normalization_668/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_668/moving_mean*
_output_shapes
:a*
dtype0
¦
'batch_normalization_668/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*8
shared_name)'batch_normalization_668/moving_variance

;batch_normalization_668/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_668/moving_variance*
_output_shapes
:a*
dtype0
|
dense_738/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:aA*!
shared_namedense_738/kernel
u
$dense_738/kernel/Read/ReadVariableOpReadVariableOpdense_738/kernel*
_output_shapes

:aA*
dtype0
t
dense_738/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*
shared_namedense_738/bias
m
"dense_738/bias/Read/ReadVariableOpReadVariableOpdense_738/bias*
_output_shapes
:A*
dtype0

batch_normalization_669/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*.
shared_namebatch_normalization_669/gamma

1batch_normalization_669/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_669/gamma*
_output_shapes
:A*
dtype0

batch_normalization_669/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*-
shared_namebatch_normalization_669/beta

0batch_normalization_669/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_669/beta*
_output_shapes
:A*
dtype0

#batch_normalization_669/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#batch_normalization_669/moving_mean

7batch_normalization_669/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_669/moving_mean*
_output_shapes
:A*
dtype0
¦
'batch_normalization_669/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*8
shared_name)'batch_normalization_669/moving_variance

;batch_normalization_669/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_669/moving_variance*
_output_shapes
:A*
dtype0
|
dense_739/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*!
shared_namedense_739/kernel
u
$dense_739/kernel/Read/ReadVariableOpReadVariableOpdense_739/kernel*
_output_shapes

:AA*
dtype0
t
dense_739/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*
shared_namedense_739/bias
m
"dense_739/bias/Read/ReadVariableOpReadVariableOpdense_739/bias*
_output_shapes
:A*
dtype0

batch_normalization_670/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*.
shared_namebatch_normalization_670/gamma

1batch_normalization_670/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_670/gamma*
_output_shapes
:A*
dtype0

batch_normalization_670/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*-
shared_namebatch_normalization_670/beta

0batch_normalization_670/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_670/beta*
_output_shapes
:A*
dtype0

#batch_normalization_670/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#batch_normalization_670/moving_mean

7batch_normalization_670/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_670/moving_mean*
_output_shapes
:A*
dtype0
¦
'batch_normalization_670/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*8
shared_name)'batch_normalization_670/moving_variance

;batch_normalization_670/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_670/moving_variance*
_output_shapes
:A*
dtype0
|
dense_740/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*!
shared_namedense_740/kernel
u
$dense_740/kernel/Read/ReadVariableOpReadVariableOpdense_740/kernel*
_output_shapes

:AA*
dtype0
t
dense_740/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*
shared_namedense_740/bias
m
"dense_740/bias/Read/ReadVariableOpReadVariableOpdense_740/bias*
_output_shapes
:A*
dtype0

batch_normalization_671/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*.
shared_namebatch_normalization_671/gamma

1batch_normalization_671/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_671/gamma*
_output_shapes
:A*
dtype0

batch_normalization_671/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*-
shared_namebatch_normalization_671/beta

0batch_normalization_671/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_671/beta*
_output_shapes
:A*
dtype0

#batch_normalization_671/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#batch_normalization_671/moving_mean

7batch_normalization_671/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_671/moving_mean*
_output_shapes
:A*
dtype0
¦
'batch_normalization_671/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*8
shared_name)'batch_normalization_671/moving_variance

;batch_normalization_671/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_671/moving_variance*
_output_shapes
:A*
dtype0
|
dense_741/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*!
shared_namedense_741/kernel
u
$dense_741/kernel/Read/ReadVariableOpReadVariableOpdense_741/kernel*
_output_shapes

:AA*
dtype0
t
dense_741/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*
shared_namedense_741/bias
m
"dense_741/bias/Read/ReadVariableOpReadVariableOpdense_741/bias*
_output_shapes
:A*
dtype0

batch_normalization_672/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*.
shared_namebatch_normalization_672/gamma

1batch_normalization_672/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_672/gamma*
_output_shapes
:A*
dtype0

batch_normalization_672/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*-
shared_namebatch_normalization_672/beta

0batch_normalization_672/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_672/beta*
_output_shapes
:A*
dtype0

#batch_normalization_672/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#batch_normalization_672/moving_mean

7batch_normalization_672/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_672/moving_mean*
_output_shapes
:A*
dtype0
¦
'batch_normalization_672/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*8
shared_name)'batch_normalization_672/moving_variance

;batch_normalization_672/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_672/moving_variance*
_output_shapes
:A*
dtype0
|
dense_742/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*!
shared_namedense_742/kernel
u
$dense_742/kernel/Read/ReadVariableOpReadVariableOpdense_742/kernel*
_output_shapes

:AA*
dtype0
t
dense_742/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*
shared_namedense_742/bias
m
"dense_742/bias/Read/ReadVariableOpReadVariableOpdense_742/bias*
_output_shapes
:A*
dtype0

batch_normalization_673/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*.
shared_namebatch_normalization_673/gamma

1batch_normalization_673/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_673/gamma*
_output_shapes
:A*
dtype0

batch_normalization_673/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*-
shared_namebatch_normalization_673/beta

0batch_normalization_673/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_673/beta*
_output_shapes
:A*
dtype0

#batch_normalization_673/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#batch_normalization_673/moving_mean

7batch_normalization_673/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_673/moving_mean*
_output_shapes
:A*
dtype0
¦
'batch_normalization_673/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*8
shared_name)'batch_normalization_673/moving_variance

;batch_normalization_673/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_673/moving_variance*
_output_shapes
:A*
dtype0
|
dense_743/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:A*!
shared_namedense_743/kernel
u
$dense_743/kernel/Read/ReadVariableOpReadVariableOpdense_743/kernel*
_output_shapes

:A*
dtype0
t
dense_743/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_743/bias
m
"dense_743/bias/Read/ReadVariableOpReadVariableOpdense_743/bias*
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
Adam/dense_734/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:S*(
shared_nameAdam/dense_734/kernel/m

+Adam/dense_734/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_734/kernel/m*
_output_shapes

:S*
dtype0

Adam/dense_734/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*&
shared_nameAdam/dense_734/bias/m
{
)Adam/dense_734/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_734/bias/m*
_output_shapes
:S*
dtype0
 
$Adam/batch_normalization_665/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*5
shared_name&$Adam/batch_normalization_665/gamma/m

8Adam/batch_normalization_665/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_665/gamma/m*
_output_shapes
:S*
dtype0

#Adam/batch_normalization_665/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*4
shared_name%#Adam/batch_normalization_665/beta/m

7Adam/batch_normalization_665/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_665/beta/m*
_output_shapes
:S*
dtype0

Adam/dense_735/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:SS*(
shared_nameAdam/dense_735/kernel/m

+Adam/dense_735/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_735/kernel/m*
_output_shapes

:SS*
dtype0

Adam/dense_735/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*&
shared_nameAdam/dense_735/bias/m
{
)Adam/dense_735/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_735/bias/m*
_output_shapes
:S*
dtype0
 
$Adam/batch_normalization_666/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*5
shared_name&$Adam/batch_normalization_666/gamma/m

8Adam/batch_normalization_666/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_666/gamma/m*
_output_shapes
:S*
dtype0

#Adam/batch_normalization_666/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*4
shared_name%#Adam/batch_normalization_666/beta/m

7Adam/batch_normalization_666/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_666/beta/m*
_output_shapes
:S*
dtype0

Adam/dense_736/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Sa*(
shared_nameAdam/dense_736/kernel/m

+Adam/dense_736/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_736/kernel/m*
_output_shapes

:Sa*
dtype0

Adam/dense_736/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*&
shared_nameAdam/dense_736/bias/m
{
)Adam/dense_736/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_736/bias/m*
_output_shapes
:a*
dtype0
 
$Adam/batch_normalization_667/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*5
shared_name&$Adam/batch_normalization_667/gamma/m

8Adam/batch_normalization_667/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_667/gamma/m*
_output_shapes
:a*
dtype0

#Adam/batch_normalization_667/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#Adam/batch_normalization_667/beta/m

7Adam/batch_normalization_667/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_667/beta/m*
_output_shapes
:a*
dtype0

Adam/dense_737/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:aa*(
shared_nameAdam/dense_737/kernel/m

+Adam/dense_737/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_737/kernel/m*
_output_shapes

:aa*
dtype0

Adam/dense_737/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*&
shared_nameAdam/dense_737/bias/m
{
)Adam/dense_737/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_737/bias/m*
_output_shapes
:a*
dtype0
 
$Adam/batch_normalization_668/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*5
shared_name&$Adam/batch_normalization_668/gamma/m

8Adam/batch_normalization_668/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_668/gamma/m*
_output_shapes
:a*
dtype0

#Adam/batch_normalization_668/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#Adam/batch_normalization_668/beta/m

7Adam/batch_normalization_668/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_668/beta/m*
_output_shapes
:a*
dtype0

Adam/dense_738/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:aA*(
shared_nameAdam/dense_738/kernel/m

+Adam/dense_738/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_738/kernel/m*
_output_shapes

:aA*
dtype0

Adam/dense_738/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_738/bias/m
{
)Adam/dense_738/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_738/bias/m*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_669/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_669/gamma/m

8Adam/batch_normalization_669/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_669/gamma/m*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_669/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_669/beta/m

7Adam/batch_normalization_669/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_669/beta/m*
_output_shapes
:A*
dtype0

Adam/dense_739/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*(
shared_nameAdam/dense_739/kernel/m

+Adam/dense_739/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_739/kernel/m*
_output_shapes

:AA*
dtype0

Adam/dense_739/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_739/bias/m
{
)Adam/dense_739/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_739/bias/m*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_670/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_670/gamma/m

8Adam/batch_normalization_670/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_670/gamma/m*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_670/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_670/beta/m

7Adam/batch_normalization_670/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_670/beta/m*
_output_shapes
:A*
dtype0

Adam/dense_740/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*(
shared_nameAdam/dense_740/kernel/m

+Adam/dense_740/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_740/kernel/m*
_output_shapes

:AA*
dtype0

Adam/dense_740/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_740/bias/m
{
)Adam/dense_740/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_740/bias/m*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_671/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_671/gamma/m

8Adam/batch_normalization_671/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_671/gamma/m*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_671/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_671/beta/m

7Adam/batch_normalization_671/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_671/beta/m*
_output_shapes
:A*
dtype0

Adam/dense_741/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*(
shared_nameAdam/dense_741/kernel/m

+Adam/dense_741/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_741/kernel/m*
_output_shapes

:AA*
dtype0

Adam/dense_741/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_741/bias/m
{
)Adam/dense_741/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_741/bias/m*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_672/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_672/gamma/m

8Adam/batch_normalization_672/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_672/gamma/m*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_672/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_672/beta/m

7Adam/batch_normalization_672/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_672/beta/m*
_output_shapes
:A*
dtype0

Adam/dense_742/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*(
shared_nameAdam/dense_742/kernel/m

+Adam/dense_742/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_742/kernel/m*
_output_shapes

:AA*
dtype0

Adam/dense_742/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_742/bias/m
{
)Adam/dense_742/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_742/bias/m*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_673/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_673/gamma/m

8Adam/batch_normalization_673/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_673/gamma/m*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_673/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_673/beta/m

7Adam/batch_normalization_673/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_673/beta/m*
_output_shapes
:A*
dtype0

Adam/dense_743/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:A*(
shared_nameAdam/dense_743/kernel/m

+Adam/dense_743/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_743/kernel/m*
_output_shapes

:A*
dtype0

Adam/dense_743/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_743/bias/m
{
)Adam/dense_743/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_743/bias/m*
_output_shapes
:*
dtype0

Adam/dense_734/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:S*(
shared_nameAdam/dense_734/kernel/v

+Adam/dense_734/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_734/kernel/v*
_output_shapes

:S*
dtype0

Adam/dense_734/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*&
shared_nameAdam/dense_734/bias/v
{
)Adam/dense_734/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_734/bias/v*
_output_shapes
:S*
dtype0
 
$Adam/batch_normalization_665/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*5
shared_name&$Adam/batch_normalization_665/gamma/v

8Adam/batch_normalization_665/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_665/gamma/v*
_output_shapes
:S*
dtype0

#Adam/batch_normalization_665/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*4
shared_name%#Adam/batch_normalization_665/beta/v

7Adam/batch_normalization_665/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_665/beta/v*
_output_shapes
:S*
dtype0

Adam/dense_735/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:SS*(
shared_nameAdam/dense_735/kernel/v

+Adam/dense_735/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_735/kernel/v*
_output_shapes

:SS*
dtype0

Adam/dense_735/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*&
shared_nameAdam/dense_735/bias/v
{
)Adam/dense_735/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_735/bias/v*
_output_shapes
:S*
dtype0
 
$Adam/batch_normalization_666/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*5
shared_name&$Adam/batch_normalization_666/gamma/v

8Adam/batch_normalization_666/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_666/gamma/v*
_output_shapes
:S*
dtype0

#Adam/batch_normalization_666/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*4
shared_name%#Adam/batch_normalization_666/beta/v

7Adam/batch_normalization_666/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_666/beta/v*
_output_shapes
:S*
dtype0

Adam/dense_736/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Sa*(
shared_nameAdam/dense_736/kernel/v

+Adam/dense_736/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_736/kernel/v*
_output_shapes

:Sa*
dtype0

Adam/dense_736/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*&
shared_nameAdam/dense_736/bias/v
{
)Adam/dense_736/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_736/bias/v*
_output_shapes
:a*
dtype0
 
$Adam/batch_normalization_667/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*5
shared_name&$Adam/batch_normalization_667/gamma/v

8Adam/batch_normalization_667/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_667/gamma/v*
_output_shapes
:a*
dtype0

#Adam/batch_normalization_667/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#Adam/batch_normalization_667/beta/v

7Adam/batch_normalization_667/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_667/beta/v*
_output_shapes
:a*
dtype0

Adam/dense_737/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:aa*(
shared_nameAdam/dense_737/kernel/v

+Adam/dense_737/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_737/kernel/v*
_output_shapes

:aa*
dtype0

Adam/dense_737/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*&
shared_nameAdam/dense_737/bias/v
{
)Adam/dense_737/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_737/bias/v*
_output_shapes
:a*
dtype0
 
$Adam/batch_normalization_668/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*5
shared_name&$Adam/batch_normalization_668/gamma/v

8Adam/batch_normalization_668/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_668/gamma/v*
_output_shapes
:a*
dtype0

#Adam/batch_normalization_668/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:a*4
shared_name%#Adam/batch_normalization_668/beta/v

7Adam/batch_normalization_668/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_668/beta/v*
_output_shapes
:a*
dtype0

Adam/dense_738/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:aA*(
shared_nameAdam/dense_738/kernel/v

+Adam/dense_738/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_738/kernel/v*
_output_shapes

:aA*
dtype0

Adam/dense_738/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_738/bias/v
{
)Adam/dense_738/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_738/bias/v*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_669/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_669/gamma/v

8Adam/batch_normalization_669/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_669/gamma/v*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_669/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_669/beta/v

7Adam/batch_normalization_669/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_669/beta/v*
_output_shapes
:A*
dtype0

Adam/dense_739/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*(
shared_nameAdam/dense_739/kernel/v

+Adam/dense_739/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_739/kernel/v*
_output_shapes

:AA*
dtype0

Adam/dense_739/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_739/bias/v
{
)Adam/dense_739/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_739/bias/v*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_670/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_670/gamma/v

8Adam/batch_normalization_670/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_670/gamma/v*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_670/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_670/beta/v

7Adam/batch_normalization_670/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_670/beta/v*
_output_shapes
:A*
dtype0

Adam/dense_740/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*(
shared_nameAdam/dense_740/kernel/v

+Adam/dense_740/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_740/kernel/v*
_output_shapes

:AA*
dtype0

Adam/dense_740/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_740/bias/v
{
)Adam/dense_740/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_740/bias/v*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_671/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_671/gamma/v

8Adam/batch_normalization_671/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_671/gamma/v*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_671/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_671/beta/v

7Adam/batch_normalization_671/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_671/beta/v*
_output_shapes
:A*
dtype0

Adam/dense_741/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*(
shared_nameAdam/dense_741/kernel/v

+Adam/dense_741/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_741/kernel/v*
_output_shapes

:AA*
dtype0

Adam/dense_741/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_741/bias/v
{
)Adam/dense_741/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_741/bias/v*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_672/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_672/gamma/v

8Adam/batch_normalization_672/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_672/gamma/v*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_672/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_672/beta/v

7Adam/batch_normalization_672/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_672/beta/v*
_output_shapes
:A*
dtype0

Adam/dense_742/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AA*(
shared_nameAdam/dense_742/kernel/v

+Adam/dense_742/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_742/kernel/v*
_output_shapes

:AA*
dtype0

Adam/dense_742/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*&
shared_nameAdam/dense_742/bias/v
{
)Adam/dense_742/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_742/bias/v*
_output_shapes
:A*
dtype0
 
$Adam/batch_normalization_673/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*5
shared_name&$Adam/batch_normalization_673/gamma/v

8Adam/batch_normalization_673/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_673/gamma/v*
_output_shapes
:A*
dtype0

#Adam/batch_normalization_673/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:A*4
shared_name%#Adam/batch_normalization_673/beta/v

7Adam/batch_normalization_673/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_673/beta/v*
_output_shapes
:A*
dtype0

Adam/dense_743/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:A*(
shared_nameAdam/dense_743/kernel/v

+Adam/dense_743/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_743/kernel/v*
_output_shapes

:A*
dtype0

Adam/dense_743/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_743/bias/v
{
)Adam/dense_743/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_743/bias/v*
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
VARIABLE_VALUEdense_734/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_734/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_665/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_665/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_665/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_665/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
VARIABLE_VALUEdense_735/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_735/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_666/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_666/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_666/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_666/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
VARIABLE_VALUEdense_736/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_736/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_667/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_667/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_667/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_667/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
VARIABLE_VALUEdense_737/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_737/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_668/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_668/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_668/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_668/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_738/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_738/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_669/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_669/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_669/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_669/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_739/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_739/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_670/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_670/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_670/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_670/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_740/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_740/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_671/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_671/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_671/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_671/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_741/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_741/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_672/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_672/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_672/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_672/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_742/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_742/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_673/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_673/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_673/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_673/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_743/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_743/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_734/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_734/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_665/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_665/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_735/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_735/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_666/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_666/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_736/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_736/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_667/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_667/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_737/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_737/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_668/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_668/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_738/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_738/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_669/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_669/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_739/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_739/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_670/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_670/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_740/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_740/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_671/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_671/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_741/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_741/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_672/gamma/mRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_672/beta/mQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_742/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_742/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_673/gamma/mRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_673/beta/mQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_743/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_743/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_734/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_734/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_665/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_665/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_735/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_735/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_666/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_666/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_736/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_736/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_667/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_667/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_737/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_737/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_668/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_668/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_738/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_738/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_669/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_669/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_739/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_739/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_670/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_670/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_740/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_740/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_671/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_671/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_741/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_741/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_672/gamma/vRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_672/beta/vQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_742/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_742/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_673/gamma/vRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_673/beta/vQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_743/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_743/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

&serving_default_normalization_69_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ú
StatefulPartitionedCallStatefulPartitionedCall&serving_default_normalization_69_inputConstConst_1dense_734/kerneldense_734/bias'batch_normalization_665/moving_variancebatch_normalization_665/gamma#batch_normalization_665/moving_meanbatch_normalization_665/betadense_735/kerneldense_735/bias'batch_normalization_666/moving_variancebatch_normalization_666/gamma#batch_normalization_666/moving_meanbatch_normalization_666/betadense_736/kerneldense_736/bias'batch_normalization_667/moving_variancebatch_normalization_667/gamma#batch_normalization_667/moving_meanbatch_normalization_667/betadense_737/kerneldense_737/bias'batch_normalization_668/moving_variancebatch_normalization_668/gamma#batch_normalization_668/moving_meanbatch_normalization_668/betadense_738/kerneldense_738/bias'batch_normalization_669/moving_variancebatch_normalization_669/gamma#batch_normalization_669/moving_meanbatch_normalization_669/betadense_739/kerneldense_739/bias'batch_normalization_670/moving_variancebatch_normalization_670/gamma#batch_normalization_670/moving_meanbatch_normalization_670/betadense_740/kerneldense_740/bias'batch_normalization_671/moving_variancebatch_normalization_671/gamma#batch_normalization_671/moving_meanbatch_normalization_671/betadense_741/kerneldense_741/bias'batch_normalization_672/moving_variancebatch_normalization_672/gamma#batch_normalization_672/moving_meanbatch_normalization_672/betadense_742/kerneldense_742/bias'batch_normalization_673/moving_variancebatch_normalization_673/gamma#batch_normalization_673/moving_meanbatch_normalization_673/betadense_743/kerneldense_743/bias*F
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
$__inference_signature_wrapper_736082
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ç8
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp$dense_734/kernel/Read/ReadVariableOp"dense_734/bias/Read/ReadVariableOp1batch_normalization_665/gamma/Read/ReadVariableOp0batch_normalization_665/beta/Read/ReadVariableOp7batch_normalization_665/moving_mean/Read/ReadVariableOp;batch_normalization_665/moving_variance/Read/ReadVariableOp$dense_735/kernel/Read/ReadVariableOp"dense_735/bias/Read/ReadVariableOp1batch_normalization_666/gamma/Read/ReadVariableOp0batch_normalization_666/beta/Read/ReadVariableOp7batch_normalization_666/moving_mean/Read/ReadVariableOp;batch_normalization_666/moving_variance/Read/ReadVariableOp$dense_736/kernel/Read/ReadVariableOp"dense_736/bias/Read/ReadVariableOp1batch_normalization_667/gamma/Read/ReadVariableOp0batch_normalization_667/beta/Read/ReadVariableOp7batch_normalization_667/moving_mean/Read/ReadVariableOp;batch_normalization_667/moving_variance/Read/ReadVariableOp$dense_737/kernel/Read/ReadVariableOp"dense_737/bias/Read/ReadVariableOp1batch_normalization_668/gamma/Read/ReadVariableOp0batch_normalization_668/beta/Read/ReadVariableOp7batch_normalization_668/moving_mean/Read/ReadVariableOp;batch_normalization_668/moving_variance/Read/ReadVariableOp$dense_738/kernel/Read/ReadVariableOp"dense_738/bias/Read/ReadVariableOp1batch_normalization_669/gamma/Read/ReadVariableOp0batch_normalization_669/beta/Read/ReadVariableOp7batch_normalization_669/moving_mean/Read/ReadVariableOp;batch_normalization_669/moving_variance/Read/ReadVariableOp$dense_739/kernel/Read/ReadVariableOp"dense_739/bias/Read/ReadVariableOp1batch_normalization_670/gamma/Read/ReadVariableOp0batch_normalization_670/beta/Read/ReadVariableOp7batch_normalization_670/moving_mean/Read/ReadVariableOp;batch_normalization_670/moving_variance/Read/ReadVariableOp$dense_740/kernel/Read/ReadVariableOp"dense_740/bias/Read/ReadVariableOp1batch_normalization_671/gamma/Read/ReadVariableOp0batch_normalization_671/beta/Read/ReadVariableOp7batch_normalization_671/moving_mean/Read/ReadVariableOp;batch_normalization_671/moving_variance/Read/ReadVariableOp$dense_741/kernel/Read/ReadVariableOp"dense_741/bias/Read/ReadVariableOp1batch_normalization_672/gamma/Read/ReadVariableOp0batch_normalization_672/beta/Read/ReadVariableOp7batch_normalization_672/moving_mean/Read/ReadVariableOp;batch_normalization_672/moving_variance/Read/ReadVariableOp$dense_742/kernel/Read/ReadVariableOp"dense_742/bias/Read/ReadVariableOp1batch_normalization_673/gamma/Read/ReadVariableOp0batch_normalization_673/beta/Read/ReadVariableOp7batch_normalization_673/moving_mean/Read/ReadVariableOp;batch_normalization_673/moving_variance/Read/ReadVariableOp$dense_743/kernel/Read/ReadVariableOp"dense_743/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_734/kernel/m/Read/ReadVariableOp)Adam/dense_734/bias/m/Read/ReadVariableOp8Adam/batch_normalization_665/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_665/beta/m/Read/ReadVariableOp+Adam/dense_735/kernel/m/Read/ReadVariableOp)Adam/dense_735/bias/m/Read/ReadVariableOp8Adam/batch_normalization_666/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_666/beta/m/Read/ReadVariableOp+Adam/dense_736/kernel/m/Read/ReadVariableOp)Adam/dense_736/bias/m/Read/ReadVariableOp8Adam/batch_normalization_667/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_667/beta/m/Read/ReadVariableOp+Adam/dense_737/kernel/m/Read/ReadVariableOp)Adam/dense_737/bias/m/Read/ReadVariableOp8Adam/batch_normalization_668/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_668/beta/m/Read/ReadVariableOp+Adam/dense_738/kernel/m/Read/ReadVariableOp)Adam/dense_738/bias/m/Read/ReadVariableOp8Adam/batch_normalization_669/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_669/beta/m/Read/ReadVariableOp+Adam/dense_739/kernel/m/Read/ReadVariableOp)Adam/dense_739/bias/m/Read/ReadVariableOp8Adam/batch_normalization_670/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_670/beta/m/Read/ReadVariableOp+Adam/dense_740/kernel/m/Read/ReadVariableOp)Adam/dense_740/bias/m/Read/ReadVariableOp8Adam/batch_normalization_671/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_671/beta/m/Read/ReadVariableOp+Adam/dense_741/kernel/m/Read/ReadVariableOp)Adam/dense_741/bias/m/Read/ReadVariableOp8Adam/batch_normalization_672/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_672/beta/m/Read/ReadVariableOp+Adam/dense_742/kernel/m/Read/ReadVariableOp)Adam/dense_742/bias/m/Read/ReadVariableOp8Adam/batch_normalization_673/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_673/beta/m/Read/ReadVariableOp+Adam/dense_743/kernel/m/Read/ReadVariableOp)Adam/dense_743/bias/m/Read/ReadVariableOp+Adam/dense_734/kernel/v/Read/ReadVariableOp)Adam/dense_734/bias/v/Read/ReadVariableOp8Adam/batch_normalization_665/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_665/beta/v/Read/ReadVariableOp+Adam/dense_735/kernel/v/Read/ReadVariableOp)Adam/dense_735/bias/v/Read/ReadVariableOp8Adam/batch_normalization_666/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_666/beta/v/Read/ReadVariableOp+Adam/dense_736/kernel/v/Read/ReadVariableOp)Adam/dense_736/bias/v/Read/ReadVariableOp8Adam/batch_normalization_667/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_667/beta/v/Read/ReadVariableOp+Adam/dense_737/kernel/v/Read/ReadVariableOp)Adam/dense_737/bias/v/Read/ReadVariableOp8Adam/batch_normalization_668/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_668/beta/v/Read/ReadVariableOp+Adam/dense_738/kernel/v/Read/ReadVariableOp)Adam/dense_738/bias/v/Read/ReadVariableOp8Adam/batch_normalization_669/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_669/beta/v/Read/ReadVariableOp+Adam/dense_739/kernel/v/Read/ReadVariableOp)Adam/dense_739/bias/v/Read/ReadVariableOp8Adam/batch_normalization_670/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_670/beta/v/Read/ReadVariableOp+Adam/dense_740/kernel/v/Read/ReadVariableOp)Adam/dense_740/bias/v/Read/ReadVariableOp8Adam/batch_normalization_671/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_671/beta/v/Read/ReadVariableOp+Adam/dense_741/kernel/v/Read/ReadVariableOp)Adam/dense_741/bias/v/Read/ReadVariableOp8Adam/batch_normalization_672/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_672/beta/v/Read/ReadVariableOp+Adam/dense_742/kernel/v/Read/ReadVariableOp)Adam/dense_742/bias/v/Read/ReadVariableOp8Adam/batch_normalization_673/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_673/beta/v/Read/ReadVariableOp+Adam/dense_743/kernel/v/Read/ReadVariableOp)Adam/dense_743/bias/v/Read/ReadVariableOpConst_2*
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
__inference__traced_save_737577
¼"
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_734/kerneldense_734/biasbatch_normalization_665/gammabatch_normalization_665/beta#batch_normalization_665/moving_mean'batch_normalization_665/moving_variancedense_735/kerneldense_735/biasbatch_normalization_666/gammabatch_normalization_666/beta#batch_normalization_666/moving_mean'batch_normalization_666/moving_variancedense_736/kerneldense_736/biasbatch_normalization_667/gammabatch_normalization_667/beta#batch_normalization_667/moving_mean'batch_normalization_667/moving_variancedense_737/kerneldense_737/biasbatch_normalization_668/gammabatch_normalization_668/beta#batch_normalization_668/moving_mean'batch_normalization_668/moving_variancedense_738/kerneldense_738/biasbatch_normalization_669/gammabatch_normalization_669/beta#batch_normalization_669/moving_mean'batch_normalization_669/moving_variancedense_739/kerneldense_739/biasbatch_normalization_670/gammabatch_normalization_670/beta#batch_normalization_670/moving_mean'batch_normalization_670/moving_variancedense_740/kerneldense_740/biasbatch_normalization_671/gammabatch_normalization_671/beta#batch_normalization_671/moving_mean'batch_normalization_671/moving_variancedense_741/kerneldense_741/biasbatch_normalization_672/gammabatch_normalization_672/beta#batch_normalization_672/moving_mean'batch_normalization_672/moving_variancedense_742/kerneldense_742/biasbatch_normalization_673/gammabatch_normalization_673/beta#batch_normalization_673/moving_mean'batch_normalization_673/moving_variancedense_743/kerneldense_743/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount_1Adam/dense_734/kernel/mAdam/dense_734/bias/m$Adam/batch_normalization_665/gamma/m#Adam/batch_normalization_665/beta/mAdam/dense_735/kernel/mAdam/dense_735/bias/m$Adam/batch_normalization_666/gamma/m#Adam/batch_normalization_666/beta/mAdam/dense_736/kernel/mAdam/dense_736/bias/m$Adam/batch_normalization_667/gamma/m#Adam/batch_normalization_667/beta/mAdam/dense_737/kernel/mAdam/dense_737/bias/m$Adam/batch_normalization_668/gamma/m#Adam/batch_normalization_668/beta/mAdam/dense_738/kernel/mAdam/dense_738/bias/m$Adam/batch_normalization_669/gamma/m#Adam/batch_normalization_669/beta/mAdam/dense_739/kernel/mAdam/dense_739/bias/m$Adam/batch_normalization_670/gamma/m#Adam/batch_normalization_670/beta/mAdam/dense_740/kernel/mAdam/dense_740/bias/m$Adam/batch_normalization_671/gamma/m#Adam/batch_normalization_671/beta/mAdam/dense_741/kernel/mAdam/dense_741/bias/m$Adam/batch_normalization_672/gamma/m#Adam/batch_normalization_672/beta/mAdam/dense_742/kernel/mAdam/dense_742/bias/m$Adam/batch_normalization_673/gamma/m#Adam/batch_normalization_673/beta/mAdam/dense_743/kernel/mAdam/dense_743/bias/mAdam/dense_734/kernel/vAdam/dense_734/bias/v$Adam/batch_normalization_665/gamma/v#Adam/batch_normalization_665/beta/vAdam/dense_735/kernel/vAdam/dense_735/bias/v$Adam/batch_normalization_666/gamma/v#Adam/batch_normalization_666/beta/vAdam/dense_736/kernel/vAdam/dense_736/bias/v$Adam/batch_normalization_667/gamma/v#Adam/batch_normalization_667/beta/vAdam/dense_737/kernel/vAdam/dense_737/bias/v$Adam/batch_normalization_668/gamma/v#Adam/batch_normalization_668/beta/vAdam/dense_738/kernel/vAdam/dense_738/bias/v$Adam/batch_normalization_669/gamma/v#Adam/batch_normalization_669/beta/vAdam/dense_739/kernel/vAdam/dense_739/bias/v$Adam/batch_normalization_670/gamma/v#Adam/batch_normalization_670/beta/vAdam/dense_740/kernel/vAdam/dense_740/bias/v$Adam/batch_normalization_671/gamma/v#Adam/batch_normalization_671/beta/vAdam/dense_741/kernel/vAdam/dense_741/bias/v$Adam/batch_normalization_672/gamma/v#Adam/batch_normalization_672/beta/vAdam/dense_742/kernel/vAdam/dense_742/bias/v$Adam/batch_normalization_673/gamma/v#Adam/batch_normalization_673/beta/vAdam/dense_743/kernel/vAdam/dense_743/bias/v*
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
"__inference__traced_restore_738010è"
Ð
²
S__inference_batch_normalization_666_layer_call_and_return_conditional_losses_736303

inputs/
!batchnorm_readvariableop_resource:S3
%batchnorm_mul_readvariableop_resource:S1
#batchnorm_readvariableop_1_resource:S1
#batchnorm_readvariableop_2_resource:S
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:S*
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
:SP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:S~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:S*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Sc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:S*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Sz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:S*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Sr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿS: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
Ä

*__inference_dense_742_layer_call_fn_737010

inputs
unknown:AA
	unknown_0:A
identity¢StatefulPartitionedCallÚ
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
GPU 2J 8 *N
fIRG
E__inference_dense_742_layer_call_and_return_conditional_losses_734011o
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
Ä

*__inference_dense_739_layer_call_fn_736683

inputs
unknown:AA
	unknown_0:A
identity¢StatefulPartitionedCallÚ
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
GPU 2J 8 *N
fIRG
E__inference_dense_739_layer_call_and_return_conditional_losses_733915o
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
%
ì
S__inference_batch_normalization_668_layer_call_and_return_conditional_losses_736555

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
È	
ö
E__inference_dense_742_layer_call_and_return_conditional_losses_734011

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
×
Ò
.__inference_sequential_69_layer_call_fn_735264

inputs
unknown
	unknown_0
	unknown_1:S
	unknown_2:S
	unknown_3:S
	unknown_4:S
	unknown_5:S
	unknown_6:S
	unknown_7:SS
	unknown_8:S
	unknown_9:S

unknown_10:S

unknown_11:S

unknown_12:S

unknown_13:Sa

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

unknown_25:aA

unknown_26:A

unknown_27:A

unknown_28:A

unknown_29:A

unknown_30:A

unknown_31:AA

unknown_32:A

unknown_33:A

unknown_34:A

unknown_35:A

unknown_36:A

unknown_37:AA

unknown_38:A

unknown_39:A

unknown_40:A

unknown_41:A

unknown_42:A

unknown_43:AA

unknown_44:A

unknown_45:A

unknown_46:A

unknown_47:A

unknown_48:A

unknown_49:AA

unknown_50:A

unknown_51:A

unknown_52:A

unknown_53:A

unknown_54:A

unknown_55:A

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
I__inference_sequential_69_layer_call_and_return_conditional_losses_734050o
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
«
L
0__inference_leaky_re_lu_670_layer_call_fn_736778

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
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_670_layer_call_and_return_conditional_losses_733935`
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
å
g
K__inference_leaky_re_lu_669_layer_call_and_return_conditional_losses_736674

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
¬
Ó
8__inference_batch_normalization_666_layer_call_fn_736270

inputs
unknown:S
	unknown_0:S
	unknown_1:S
	unknown_2:S
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_666_layer_call_and_return_conditional_losses_733099o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿS: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_672_layer_call_and_return_conditional_losses_736991

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
«
L
0__inference_leaky_re_lu_672_layer_call_fn_736996

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
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_672_layer_call_and_return_conditional_losses_733999`
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
È	
ö
E__inference_dense_735_layer_call_and_return_conditional_losses_736257

inputs0
matmul_readvariableop_resource:SS-
biasadd_readvariableop_resource:S
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:SS*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:S*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿS: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_667_layer_call_and_return_conditional_losses_736456

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
%
ì
S__inference_batch_normalization_665_layer_call_and_return_conditional_losses_736228

inputs5
'assignmovingavg_readvariableop_resource:S7
)assignmovingavg_1_readvariableop_resource:S3
%batchnorm_mul_readvariableop_resource:S/
!batchnorm_readvariableop_resource:S
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:S*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:S
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:S*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:S*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:S*
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
:S*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Sx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:S¬
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
:S*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:S~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:S´
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
:SP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:S~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:S*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Sc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Sv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:S*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Sr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿS: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_670_layer_call_fn_736706

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_670_layer_call_and_return_conditional_losses_733427o
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
È	
ö
E__inference_dense_737_layer_call_and_return_conditional_losses_736475

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
Ð
²
S__inference_batch_normalization_668_layer_call_and_return_conditional_losses_736521

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
0__inference_leaky_re_lu_667_layer_call_fn_736451

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
K__inference_leaky_re_lu_667_layer_call_and_return_conditional_losses_733839`
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
K__inference_leaky_re_lu_672_layer_call_and_return_conditional_losses_737001

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
ª
Ó
8__inference_batch_normalization_669_layer_call_fn_736610

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_669_layer_call_and_return_conditional_losses_733392o
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
Ä

*__inference_dense_740_layer_call_fn_736792

inputs
unknown:AA
	unknown_0:A
identity¢StatefulPartitionedCallÚ
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
GPU 2J 8 *N
fIRG
E__inference_dense_740_layer_call_and_return_conditional_losses_733947o
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
¬
Ó
8__inference_batch_normalization_665_layer_call_fn_736161

inputs
unknown:S
	unknown_0:S
	unknown_1:S
	unknown_2:S
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_665_layer_call_and_return_conditional_losses_733017o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿS: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
Û'
Ò
__inference_adapt_step_736129
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
È	
ö
E__inference_dense_740_layer_call_and_return_conditional_losses_736802

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
%
ì
S__inference_batch_normalization_673_layer_call_and_return_conditional_losses_737100

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
%
ì
S__inference_batch_normalization_668_layer_call_and_return_conditional_losses_733310

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
K__inference_leaky_re_lu_673_layer_call_and_return_conditional_losses_737110

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
È	
ö
E__inference_dense_741_layer_call_and_return_conditional_losses_733979

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
Ð
²
S__inference_batch_normalization_671_layer_call_and_return_conditional_losses_736848

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
ª
Ó
8__inference_batch_normalization_673_layer_call_fn_737046

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_673_layer_call_and_return_conditional_losses_733720o
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
¶¾
Í^
"__inference__traced_restore_738010
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 5
#assignvariableop_3_dense_734_kernel:S/
!assignvariableop_4_dense_734_bias:S>
0assignvariableop_5_batch_normalization_665_gamma:S=
/assignvariableop_6_batch_normalization_665_beta:SD
6assignvariableop_7_batch_normalization_665_moving_mean:SH
:assignvariableop_8_batch_normalization_665_moving_variance:S5
#assignvariableop_9_dense_735_kernel:SS0
"assignvariableop_10_dense_735_bias:S?
1assignvariableop_11_batch_normalization_666_gamma:S>
0assignvariableop_12_batch_normalization_666_beta:SE
7assignvariableop_13_batch_normalization_666_moving_mean:SI
;assignvariableop_14_batch_normalization_666_moving_variance:S6
$assignvariableop_15_dense_736_kernel:Sa0
"assignvariableop_16_dense_736_bias:a?
1assignvariableop_17_batch_normalization_667_gamma:a>
0assignvariableop_18_batch_normalization_667_beta:aE
7assignvariableop_19_batch_normalization_667_moving_mean:aI
;assignvariableop_20_batch_normalization_667_moving_variance:a6
$assignvariableop_21_dense_737_kernel:aa0
"assignvariableop_22_dense_737_bias:a?
1assignvariableop_23_batch_normalization_668_gamma:a>
0assignvariableop_24_batch_normalization_668_beta:aE
7assignvariableop_25_batch_normalization_668_moving_mean:aI
;assignvariableop_26_batch_normalization_668_moving_variance:a6
$assignvariableop_27_dense_738_kernel:aA0
"assignvariableop_28_dense_738_bias:A?
1assignvariableop_29_batch_normalization_669_gamma:A>
0assignvariableop_30_batch_normalization_669_beta:AE
7assignvariableop_31_batch_normalization_669_moving_mean:AI
;assignvariableop_32_batch_normalization_669_moving_variance:A6
$assignvariableop_33_dense_739_kernel:AA0
"assignvariableop_34_dense_739_bias:A?
1assignvariableop_35_batch_normalization_670_gamma:A>
0assignvariableop_36_batch_normalization_670_beta:AE
7assignvariableop_37_batch_normalization_670_moving_mean:AI
;assignvariableop_38_batch_normalization_670_moving_variance:A6
$assignvariableop_39_dense_740_kernel:AA0
"assignvariableop_40_dense_740_bias:A?
1assignvariableop_41_batch_normalization_671_gamma:A>
0assignvariableop_42_batch_normalization_671_beta:AE
7assignvariableop_43_batch_normalization_671_moving_mean:AI
;assignvariableop_44_batch_normalization_671_moving_variance:A6
$assignvariableop_45_dense_741_kernel:AA0
"assignvariableop_46_dense_741_bias:A?
1assignvariableop_47_batch_normalization_672_gamma:A>
0assignvariableop_48_batch_normalization_672_beta:AE
7assignvariableop_49_batch_normalization_672_moving_mean:AI
;assignvariableop_50_batch_normalization_672_moving_variance:A6
$assignvariableop_51_dense_742_kernel:AA0
"assignvariableop_52_dense_742_bias:A?
1assignvariableop_53_batch_normalization_673_gamma:A>
0assignvariableop_54_batch_normalization_673_beta:AE
7assignvariableop_55_batch_normalization_673_moving_mean:AI
;assignvariableop_56_batch_normalization_673_moving_variance:A6
$assignvariableop_57_dense_743_kernel:A0
"assignvariableop_58_dense_743_bias:'
assignvariableop_59_adam_iter:	 )
assignvariableop_60_adam_beta_1: )
assignvariableop_61_adam_beta_2: (
assignvariableop_62_adam_decay: #
assignvariableop_63_total: %
assignvariableop_64_count_1: =
+assignvariableop_65_adam_dense_734_kernel_m:S7
)assignvariableop_66_adam_dense_734_bias_m:SF
8assignvariableop_67_adam_batch_normalization_665_gamma_m:SE
7assignvariableop_68_adam_batch_normalization_665_beta_m:S=
+assignvariableop_69_adam_dense_735_kernel_m:SS7
)assignvariableop_70_adam_dense_735_bias_m:SF
8assignvariableop_71_adam_batch_normalization_666_gamma_m:SE
7assignvariableop_72_adam_batch_normalization_666_beta_m:S=
+assignvariableop_73_adam_dense_736_kernel_m:Sa7
)assignvariableop_74_adam_dense_736_bias_m:aF
8assignvariableop_75_adam_batch_normalization_667_gamma_m:aE
7assignvariableop_76_adam_batch_normalization_667_beta_m:a=
+assignvariableop_77_adam_dense_737_kernel_m:aa7
)assignvariableop_78_adam_dense_737_bias_m:aF
8assignvariableop_79_adam_batch_normalization_668_gamma_m:aE
7assignvariableop_80_adam_batch_normalization_668_beta_m:a=
+assignvariableop_81_adam_dense_738_kernel_m:aA7
)assignvariableop_82_adam_dense_738_bias_m:AF
8assignvariableop_83_adam_batch_normalization_669_gamma_m:AE
7assignvariableop_84_adam_batch_normalization_669_beta_m:A=
+assignvariableop_85_adam_dense_739_kernel_m:AA7
)assignvariableop_86_adam_dense_739_bias_m:AF
8assignvariableop_87_adam_batch_normalization_670_gamma_m:AE
7assignvariableop_88_adam_batch_normalization_670_beta_m:A=
+assignvariableop_89_adam_dense_740_kernel_m:AA7
)assignvariableop_90_adam_dense_740_bias_m:AF
8assignvariableop_91_adam_batch_normalization_671_gamma_m:AE
7assignvariableop_92_adam_batch_normalization_671_beta_m:A=
+assignvariableop_93_adam_dense_741_kernel_m:AA7
)assignvariableop_94_adam_dense_741_bias_m:AF
8assignvariableop_95_adam_batch_normalization_672_gamma_m:AE
7assignvariableop_96_adam_batch_normalization_672_beta_m:A=
+assignvariableop_97_adam_dense_742_kernel_m:AA7
)assignvariableop_98_adam_dense_742_bias_m:AF
8assignvariableop_99_adam_batch_normalization_673_gamma_m:AF
8assignvariableop_100_adam_batch_normalization_673_beta_m:A>
,assignvariableop_101_adam_dense_743_kernel_m:A8
*assignvariableop_102_adam_dense_743_bias_m:>
,assignvariableop_103_adam_dense_734_kernel_v:S8
*assignvariableop_104_adam_dense_734_bias_v:SG
9assignvariableop_105_adam_batch_normalization_665_gamma_v:SF
8assignvariableop_106_adam_batch_normalization_665_beta_v:S>
,assignvariableop_107_adam_dense_735_kernel_v:SS8
*assignvariableop_108_adam_dense_735_bias_v:SG
9assignvariableop_109_adam_batch_normalization_666_gamma_v:SF
8assignvariableop_110_adam_batch_normalization_666_beta_v:S>
,assignvariableop_111_adam_dense_736_kernel_v:Sa8
*assignvariableop_112_adam_dense_736_bias_v:aG
9assignvariableop_113_adam_batch_normalization_667_gamma_v:aF
8assignvariableop_114_adam_batch_normalization_667_beta_v:a>
,assignvariableop_115_adam_dense_737_kernel_v:aa8
*assignvariableop_116_adam_dense_737_bias_v:aG
9assignvariableop_117_adam_batch_normalization_668_gamma_v:aF
8assignvariableop_118_adam_batch_normalization_668_beta_v:a>
,assignvariableop_119_adam_dense_738_kernel_v:aA8
*assignvariableop_120_adam_dense_738_bias_v:AG
9assignvariableop_121_adam_batch_normalization_669_gamma_v:AF
8assignvariableop_122_adam_batch_normalization_669_beta_v:A>
,assignvariableop_123_adam_dense_739_kernel_v:AA8
*assignvariableop_124_adam_dense_739_bias_v:AG
9assignvariableop_125_adam_batch_normalization_670_gamma_v:AF
8assignvariableop_126_adam_batch_normalization_670_beta_v:A>
,assignvariableop_127_adam_dense_740_kernel_v:AA8
*assignvariableop_128_adam_dense_740_bias_v:AG
9assignvariableop_129_adam_batch_normalization_671_gamma_v:AF
8assignvariableop_130_adam_batch_normalization_671_beta_v:A>
,assignvariableop_131_adam_dense_741_kernel_v:AA8
*assignvariableop_132_adam_dense_741_bias_v:AG
9assignvariableop_133_adam_batch_normalization_672_gamma_v:AF
8assignvariableop_134_adam_batch_normalization_672_beta_v:A>
,assignvariableop_135_adam_dense_742_kernel_v:AA8
*assignvariableop_136_adam_dense_742_bias_v:AG
9assignvariableop_137_adam_batch_normalization_673_gamma_v:AF
8assignvariableop_138_adam_batch_normalization_673_beta_v:A>
,assignvariableop_139_adam_dense_743_kernel_v:A8
*assignvariableop_140_adam_dense_743_bias_v:
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
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_734_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_734_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp0assignvariableop_5_batch_normalization_665_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_665_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_665_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_8AssignVariableOp:assignvariableop_8_batch_normalization_665_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_735_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_735_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_11AssignVariableOp1assignvariableop_11_batch_normalization_666_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_666_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_666_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_14AssignVariableOp;assignvariableop_14_batch_normalization_666_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_736_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_736_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_17AssignVariableOp1assignvariableop_17_batch_normalization_667_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_667_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_19AssignVariableOp7assignvariableop_19_batch_normalization_667_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_20AssignVariableOp;assignvariableop_20_batch_normalization_667_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp$assignvariableop_21_dense_737_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_737_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_23AssignVariableOp1assignvariableop_23_batch_normalization_668_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_668_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_25AssignVariableOp7assignvariableop_25_batch_normalization_668_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_26AssignVariableOp;assignvariableop_26_batch_normalization_668_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp$assignvariableop_27_dense_738_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_738_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_29AssignVariableOp1assignvariableop_29_batch_normalization_669_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_normalization_669_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_31AssignVariableOp7assignvariableop_31_batch_normalization_669_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_32AssignVariableOp;assignvariableop_32_batch_normalization_669_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp$assignvariableop_33_dense_739_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_739_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_35AssignVariableOp1assignvariableop_35_batch_normalization_670_gammaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_670_betaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_37AssignVariableOp7assignvariableop_37_batch_normalization_670_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_38AssignVariableOp;assignvariableop_38_batch_normalization_670_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp$assignvariableop_39_dense_740_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp"assignvariableop_40_dense_740_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_41AssignVariableOp1assignvariableop_41_batch_normalization_671_gammaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_42AssignVariableOp0assignvariableop_42_batch_normalization_671_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_43AssignVariableOp7assignvariableop_43_batch_normalization_671_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_44AssignVariableOp;assignvariableop_44_batch_normalization_671_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp$assignvariableop_45_dense_741_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp"assignvariableop_46_dense_741_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_47AssignVariableOp1assignvariableop_47_batch_normalization_672_gammaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_48AssignVariableOp0assignvariableop_48_batch_normalization_672_betaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_49AssignVariableOp7assignvariableop_49_batch_normalization_672_moving_meanIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_50AssignVariableOp;assignvariableop_50_batch_normalization_672_moving_varianceIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp$assignvariableop_51_dense_742_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp"assignvariableop_52_dense_742_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_53AssignVariableOp1assignvariableop_53_batch_normalization_673_gammaIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_54AssignVariableOp0assignvariableop_54_batch_normalization_673_betaIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_55AssignVariableOp7assignvariableop_55_batch_normalization_673_moving_meanIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_56AssignVariableOp;assignvariableop_56_batch_normalization_673_moving_varianceIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp$assignvariableop_57_dense_743_kernelIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp"assignvariableop_58_dense_743_biasIdentity_58:output:0"/device:CPU:0*
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
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_dense_734_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_dense_734_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_67AssignVariableOp8assignvariableop_67_adam_batch_normalization_665_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_68AssignVariableOp7assignvariableop_68_adam_batch_normalization_665_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_dense_735_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_dense_735_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_71AssignVariableOp8assignvariableop_71_adam_batch_normalization_666_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_72AssignVariableOp7assignvariableop_72_adam_batch_normalization_666_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_dense_736_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_dense_736_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_75AssignVariableOp8assignvariableop_75_adam_batch_normalization_667_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_76AssignVariableOp7assignvariableop_76_adam_batch_normalization_667_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_dense_737_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_dense_737_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_79AssignVariableOp8assignvariableop_79_adam_batch_normalization_668_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_80AssignVariableOp7assignvariableop_80_adam_batch_normalization_668_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_dense_738_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_dense_738_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_83AssignVariableOp8assignvariableop_83_adam_batch_normalization_669_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_84AssignVariableOp7assignvariableop_84_adam_batch_normalization_669_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_dense_739_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_dense_739_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_87AssignVariableOp8assignvariableop_87_adam_batch_normalization_670_gamma_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_88AssignVariableOp7assignvariableop_88_adam_batch_normalization_670_beta_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_dense_740_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_dense_740_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_91AssignVariableOp8assignvariableop_91_adam_batch_normalization_671_gamma_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_92AssignVariableOp7assignvariableop_92_adam_batch_normalization_671_beta_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_dense_741_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_dense_741_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_95AssignVariableOp8assignvariableop_95_adam_batch_normalization_672_gamma_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_96AssignVariableOp7assignvariableop_96_adam_batch_normalization_672_beta_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_dense_742_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_dense_742_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_99AssignVariableOp8assignvariableop_99_adam_batch_normalization_673_gamma_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_100AssignVariableOp8assignvariableop_100_adam_batch_normalization_673_beta_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_dense_743_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_dense_743_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_103AssignVariableOp,assignvariableop_103_adam_dense_734_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_104AssignVariableOp*assignvariableop_104_adam_dense_734_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_105AssignVariableOp9assignvariableop_105_adam_batch_normalization_665_gamma_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_106AssignVariableOp8assignvariableop_106_adam_batch_normalization_665_beta_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_107AssignVariableOp,assignvariableop_107_adam_dense_735_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_108AssignVariableOp*assignvariableop_108_adam_dense_735_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_109AssignVariableOp9assignvariableop_109_adam_batch_normalization_666_gamma_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_110AssignVariableOp8assignvariableop_110_adam_batch_normalization_666_beta_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_111AssignVariableOp,assignvariableop_111_adam_dense_736_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_112AssignVariableOp*assignvariableop_112_adam_dense_736_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_113AssignVariableOp9assignvariableop_113_adam_batch_normalization_667_gamma_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_114AssignVariableOp8assignvariableop_114_adam_batch_normalization_667_beta_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_115AssignVariableOp,assignvariableop_115_adam_dense_737_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_116AssignVariableOp*assignvariableop_116_adam_dense_737_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_117AssignVariableOp9assignvariableop_117_adam_batch_normalization_668_gamma_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_118AssignVariableOp8assignvariableop_118_adam_batch_normalization_668_beta_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_119AssignVariableOp,assignvariableop_119_adam_dense_738_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_120AssignVariableOp*assignvariableop_120_adam_dense_738_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_121AssignVariableOp9assignvariableop_121_adam_batch_normalization_669_gamma_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_122AssignVariableOp8assignvariableop_122_adam_batch_normalization_669_beta_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_123AssignVariableOp,assignvariableop_123_adam_dense_739_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_124AssignVariableOp*assignvariableop_124_adam_dense_739_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_125AssignVariableOp9assignvariableop_125_adam_batch_normalization_670_gamma_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_126AssignVariableOp8assignvariableop_126_adam_batch_normalization_670_beta_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_127AssignVariableOp,assignvariableop_127_adam_dense_740_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_128AssignVariableOp*assignvariableop_128_adam_dense_740_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_129AssignVariableOp9assignvariableop_129_adam_batch_normalization_671_gamma_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_130AssignVariableOp8assignvariableop_130_adam_batch_normalization_671_beta_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_131AssignVariableOp,assignvariableop_131_adam_dense_741_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_132AssignVariableOp*assignvariableop_132_adam_dense_741_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_133AssignVariableOp9assignvariableop_133_adam_batch_normalization_672_gamma_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_134AssignVariableOp8assignvariableop_134_adam_batch_normalization_672_beta_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_135AssignVariableOp,assignvariableop_135_adam_dense_742_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_136AssignVariableOp*assignvariableop_136_adam_dense_742_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_137AssignVariableOp9assignvariableop_137_adam_batch_normalization_673_gamma_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_138AssignVariableOp8assignvariableop_138_adam_batch_normalization_673_beta_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_139AssignVariableOp,assignvariableop_139_adam_dense_743_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_140AssignVariableOp*assignvariableop_140_adam_dense_743_bias_vIdentity_140:output:0"/device:CPU:0*
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
Å
Ò
.__inference_sequential_69_layer_call_fn_735385

inputs
unknown
	unknown_0
	unknown_1:S
	unknown_2:S
	unknown_3:S
	unknown_4:S
	unknown_5:S
	unknown_6:S
	unknown_7:SS
	unknown_8:S
	unknown_9:S

unknown_10:S

unknown_11:S

unknown_12:S

unknown_13:Sa

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

unknown_25:aA

unknown_26:A

unknown_27:A

unknown_28:A

unknown_29:A

unknown_30:A

unknown_31:AA

unknown_32:A

unknown_33:A

unknown_34:A

unknown_35:A

unknown_36:A

unknown_37:AA

unknown_38:A

unknown_39:A

unknown_40:A

unknown_41:A

unknown_42:A

unknown_43:AA

unknown_44:A

unknown_45:A

unknown_46:A

unknown_47:A

unknown_48:A

unknown_49:AA

unknown_50:A

unknown_51:A

unknown_52:A

unknown_53:A

unknown_54:A

unknown_55:A

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
I__inference_sequential_69_layer_call_and_return_conditional_losses_734597o
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
È	
ö
E__inference_dense_743_layer_call_and_return_conditional_losses_734043

inputs0
matmul_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:A*
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
:ÿÿÿÿÿÿÿÿÿA: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_666_layer_call_fn_736283

inputs
unknown:S
	unknown_0:S
	unknown_1:S
	unknown_2:S
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_666_layer_call_and_return_conditional_losses_733146o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿS: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_667_layer_call_and_return_conditional_losses_736446

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
8__inference_batch_normalization_673_layer_call_fn_737033

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_673_layer_call_and_return_conditional_losses_733673o
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
%
ì
S__inference_batch_normalization_670_layer_call_and_return_conditional_losses_736773

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
Ä

*__inference_dense_743_layer_call_fn_737119

inputs
unknown:A
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
E__inference_dense_743_layer_call_and_return_conditional_losses_734043o
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
:ÿÿÿÿÿÿÿÿÿA: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_672_layer_call_and_return_conditional_losses_733591

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
¬
Ó
8__inference_batch_normalization_672_layer_call_fn_736924

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_672_layer_call_and_return_conditional_losses_733591o
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
%
ì
S__inference_batch_normalization_665_layer_call_and_return_conditional_losses_733064

inputs5
'assignmovingavg_readvariableop_resource:S7
)assignmovingavg_1_readvariableop_resource:S3
%batchnorm_mul_readvariableop_resource:S/
!batchnorm_readvariableop_resource:S
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:S*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:S
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:S*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:S*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:S*
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
:S*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Sx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:S¬
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
:S*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:S~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:S´
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
:SP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:S~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:S*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Sc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Sv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:S*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Sr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿS: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_666_layer_call_and_return_conditional_losses_733807

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿS:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_671_layer_call_fn_736887

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
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_671_layer_call_and_return_conditional_losses_733967`
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
È	
ö
E__inference_dense_736_layer_call_and_return_conditional_losses_736366

inputs0
matmul_readvariableop_resource:Sa-
biasadd_readvariableop_resource:a
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Sa*
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
:ÿÿÿÿÿÿÿÿÿS: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_667_layer_call_fn_736379

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
S__inference_batch_normalization_667_layer_call_and_return_conditional_losses_733181o
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
S__inference_batch_normalization_669_layer_call_and_return_conditional_losses_736664

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
«
L
0__inference_leaky_re_lu_669_layer_call_fn_736669

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
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_669_layer_call_and_return_conditional_losses_733903`
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
Ä

*__inference_dense_741_layer_call_fn_736901

inputs
unknown:AA
	unknown_0:A
identity¢StatefulPartitionedCallÚ
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
GPU 2J 8 *N
fIRG
E__inference_dense_741_layer_call_and_return_conditional_losses_733979o
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
È	
ö
E__inference_dense_743_layer_call_and_return_conditional_losses_737129

inputs0
matmul_readvariableop_resource:A-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:A*
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
:ÿÿÿÿÿÿÿÿÿA: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_670_layer_call_and_return_conditional_losses_736739

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
È	
ö
E__inference_dense_741_layer_call_and_return_conditional_losses_736911

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
Ä

*__inference_dense_737_layer_call_fn_736465

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
E__inference_dense_737_layer_call_and_return_conditional_losses_733851o
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
Ä

*__inference_dense_734_layer_call_fn_736138

inputs
unknown:S
	unknown_0:S
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_734_layer_call_and_return_conditional_losses_733755o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS`
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
%
ì
S__inference_batch_normalization_672_layer_call_and_return_conditional_losses_733638

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

ë
I__inference_sequential_69_layer_call_and_return_conditional_losses_734597

inputs
normalization_69_sub_y
normalization_69_sqrt_x"
dense_734_734456:S
dense_734_734458:S,
batch_normalization_665_734461:S,
batch_normalization_665_734463:S,
batch_normalization_665_734465:S,
batch_normalization_665_734467:S"
dense_735_734471:SS
dense_735_734473:S,
batch_normalization_666_734476:S,
batch_normalization_666_734478:S,
batch_normalization_666_734480:S,
batch_normalization_666_734482:S"
dense_736_734486:Sa
dense_736_734488:a,
batch_normalization_667_734491:a,
batch_normalization_667_734493:a,
batch_normalization_667_734495:a,
batch_normalization_667_734497:a"
dense_737_734501:aa
dense_737_734503:a,
batch_normalization_668_734506:a,
batch_normalization_668_734508:a,
batch_normalization_668_734510:a,
batch_normalization_668_734512:a"
dense_738_734516:aA
dense_738_734518:A,
batch_normalization_669_734521:A,
batch_normalization_669_734523:A,
batch_normalization_669_734525:A,
batch_normalization_669_734527:A"
dense_739_734531:AA
dense_739_734533:A,
batch_normalization_670_734536:A,
batch_normalization_670_734538:A,
batch_normalization_670_734540:A,
batch_normalization_670_734542:A"
dense_740_734546:AA
dense_740_734548:A,
batch_normalization_671_734551:A,
batch_normalization_671_734553:A,
batch_normalization_671_734555:A,
batch_normalization_671_734557:A"
dense_741_734561:AA
dense_741_734563:A,
batch_normalization_672_734566:A,
batch_normalization_672_734568:A,
batch_normalization_672_734570:A,
batch_normalization_672_734572:A"
dense_742_734576:AA
dense_742_734578:A,
batch_normalization_673_734581:A,
batch_normalization_673_734583:A,
batch_normalization_673_734585:A,
batch_normalization_673_734587:A"
dense_743_734591:A
dense_743_734593:
identity¢/batch_normalization_665/StatefulPartitionedCall¢/batch_normalization_666/StatefulPartitionedCall¢/batch_normalization_667/StatefulPartitionedCall¢/batch_normalization_668/StatefulPartitionedCall¢/batch_normalization_669/StatefulPartitionedCall¢/batch_normalization_670/StatefulPartitionedCall¢/batch_normalization_671/StatefulPartitionedCall¢/batch_normalization_672/StatefulPartitionedCall¢/batch_normalization_673/StatefulPartitionedCall¢!dense_734/StatefulPartitionedCall¢!dense_735/StatefulPartitionedCall¢!dense_736/StatefulPartitionedCall¢!dense_737/StatefulPartitionedCall¢!dense_738/StatefulPartitionedCall¢!dense_739/StatefulPartitionedCall¢!dense_740/StatefulPartitionedCall¢!dense_741/StatefulPartitionedCall¢!dense_742/StatefulPartitionedCall¢!dense_743/StatefulPartitionedCallm
normalization_69/subSubinputsnormalization_69_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_69/SqrtSqrtnormalization_69_sqrt_x*
T0*
_output_shapes

:_
normalization_69/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_69/MaximumMaximumnormalization_69/Sqrt:y:0#normalization_69/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_69/truedivRealDivnormalization_69/sub:z:0normalization_69/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_734/StatefulPartitionedCallStatefulPartitionedCallnormalization_69/truediv:z:0dense_734_734456dense_734_734458*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_734_layer_call_and_return_conditional_losses_733755
/batch_normalization_665/StatefulPartitionedCallStatefulPartitionedCall*dense_734/StatefulPartitionedCall:output:0batch_normalization_665_734461batch_normalization_665_734463batch_normalization_665_734465batch_normalization_665_734467*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_665_layer_call_and_return_conditional_losses_733064ø
leaky_re_lu_665/PartitionedCallPartitionedCall8batch_normalization_665/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_665_layer_call_and_return_conditional_losses_733775
!dense_735/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_665/PartitionedCall:output:0dense_735_734471dense_735_734473*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_735_layer_call_and_return_conditional_losses_733787
/batch_normalization_666/StatefulPartitionedCallStatefulPartitionedCall*dense_735/StatefulPartitionedCall:output:0batch_normalization_666_734476batch_normalization_666_734478batch_normalization_666_734480batch_normalization_666_734482*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_666_layer_call_and_return_conditional_losses_733146ø
leaky_re_lu_666/PartitionedCallPartitionedCall8batch_normalization_666/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_666_layer_call_and_return_conditional_losses_733807
!dense_736/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_666/PartitionedCall:output:0dense_736_734486dense_736_734488*
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
E__inference_dense_736_layer_call_and_return_conditional_losses_733819
/batch_normalization_667/StatefulPartitionedCallStatefulPartitionedCall*dense_736/StatefulPartitionedCall:output:0batch_normalization_667_734491batch_normalization_667_734493batch_normalization_667_734495batch_normalization_667_734497*
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
S__inference_batch_normalization_667_layer_call_and_return_conditional_losses_733228ø
leaky_re_lu_667/PartitionedCallPartitionedCall8batch_normalization_667/StatefulPartitionedCall:output:0*
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
K__inference_leaky_re_lu_667_layer_call_and_return_conditional_losses_733839
!dense_737/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_667/PartitionedCall:output:0dense_737_734501dense_737_734503*
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
E__inference_dense_737_layer_call_and_return_conditional_losses_733851
/batch_normalization_668/StatefulPartitionedCallStatefulPartitionedCall*dense_737/StatefulPartitionedCall:output:0batch_normalization_668_734506batch_normalization_668_734508batch_normalization_668_734510batch_normalization_668_734512*
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
S__inference_batch_normalization_668_layer_call_and_return_conditional_losses_733310ø
leaky_re_lu_668/PartitionedCallPartitionedCall8batch_normalization_668/StatefulPartitionedCall:output:0*
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
K__inference_leaky_re_lu_668_layer_call_and_return_conditional_losses_733871
!dense_738/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_668/PartitionedCall:output:0dense_738_734516dense_738_734518*
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
GPU 2J 8 *N
fIRG
E__inference_dense_738_layer_call_and_return_conditional_losses_733883
/batch_normalization_669/StatefulPartitionedCallStatefulPartitionedCall*dense_738/StatefulPartitionedCall:output:0batch_normalization_669_734521batch_normalization_669_734523batch_normalization_669_734525batch_normalization_669_734527*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_669_layer_call_and_return_conditional_losses_733392ø
leaky_re_lu_669/PartitionedCallPartitionedCall8batch_normalization_669/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_669_layer_call_and_return_conditional_losses_733903
!dense_739/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_669/PartitionedCall:output:0dense_739_734531dense_739_734533*
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
GPU 2J 8 *N
fIRG
E__inference_dense_739_layer_call_and_return_conditional_losses_733915
/batch_normalization_670/StatefulPartitionedCallStatefulPartitionedCall*dense_739/StatefulPartitionedCall:output:0batch_normalization_670_734536batch_normalization_670_734538batch_normalization_670_734540batch_normalization_670_734542*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_670_layer_call_and_return_conditional_losses_733474ø
leaky_re_lu_670/PartitionedCallPartitionedCall8batch_normalization_670/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_670_layer_call_and_return_conditional_losses_733935
!dense_740/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_670/PartitionedCall:output:0dense_740_734546dense_740_734548*
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
GPU 2J 8 *N
fIRG
E__inference_dense_740_layer_call_and_return_conditional_losses_733947
/batch_normalization_671/StatefulPartitionedCallStatefulPartitionedCall*dense_740/StatefulPartitionedCall:output:0batch_normalization_671_734551batch_normalization_671_734553batch_normalization_671_734555batch_normalization_671_734557*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_671_layer_call_and_return_conditional_losses_733556ø
leaky_re_lu_671/PartitionedCallPartitionedCall8batch_normalization_671/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_671_layer_call_and_return_conditional_losses_733967
!dense_741/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_671/PartitionedCall:output:0dense_741_734561dense_741_734563*
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
GPU 2J 8 *N
fIRG
E__inference_dense_741_layer_call_and_return_conditional_losses_733979
/batch_normalization_672/StatefulPartitionedCallStatefulPartitionedCall*dense_741/StatefulPartitionedCall:output:0batch_normalization_672_734566batch_normalization_672_734568batch_normalization_672_734570batch_normalization_672_734572*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_672_layer_call_and_return_conditional_losses_733638ø
leaky_re_lu_672/PartitionedCallPartitionedCall8batch_normalization_672/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_672_layer_call_and_return_conditional_losses_733999
!dense_742/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_672/PartitionedCall:output:0dense_742_734576dense_742_734578*
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
GPU 2J 8 *N
fIRG
E__inference_dense_742_layer_call_and_return_conditional_losses_734011
/batch_normalization_673/StatefulPartitionedCallStatefulPartitionedCall*dense_742/StatefulPartitionedCall:output:0batch_normalization_673_734581batch_normalization_673_734583batch_normalization_673_734585batch_normalization_673_734587*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_673_layer_call_and_return_conditional_losses_733720ø
leaky_re_lu_673/PartitionedCallPartitionedCall8batch_normalization_673/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_673_layer_call_and_return_conditional_losses_734031
!dense_743/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_673/PartitionedCall:output:0dense_743_734591dense_743_734593*
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
E__inference_dense_743_layer_call_and_return_conditional_losses_734043y
IdentityIdentity*dense_743/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp0^batch_normalization_665/StatefulPartitionedCall0^batch_normalization_666/StatefulPartitionedCall0^batch_normalization_667/StatefulPartitionedCall0^batch_normalization_668/StatefulPartitionedCall0^batch_normalization_669/StatefulPartitionedCall0^batch_normalization_670/StatefulPartitionedCall0^batch_normalization_671/StatefulPartitionedCall0^batch_normalization_672/StatefulPartitionedCall0^batch_normalization_673/StatefulPartitionedCall"^dense_734/StatefulPartitionedCall"^dense_735/StatefulPartitionedCall"^dense_736/StatefulPartitionedCall"^dense_737/StatefulPartitionedCall"^dense_738/StatefulPartitionedCall"^dense_739/StatefulPartitionedCall"^dense_740/StatefulPartitionedCall"^dense_741/StatefulPartitionedCall"^dense_742/StatefulPartitionedCall"^dense_743/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_665/StatefulPartitionedCall/batch_normalization_665/StatefulPartitionedCall2b
/batch_normalization_666/StatefulPartitionedCall/batch_normalization_666/StatefulPartitionedCall2b
/batch_normalization_667/StatefulPartitionedCall/batch_normalization_667/StatefulPartitionedCall2b
/batch_normalization_668/StatefulPartitionedCall/batch_normalization_668/StatefulPartitionedCall2b
/batch_normalization_669/StatefulPartitionedCall/batch_normalization_669/StatefulPartitionedCall2b
/batch_normalization_670/StatefulPartitionedCall/batch_normalization_670/StatefulPartitionedCall2b
/batch_normalization_671/StatefulPartitionedCall/batch_normalization_671/StatefulPartitionedCall2b
/batch_normalization_672/StatefulPartitionedCall/batch_normalization_672/StatefulPartitionedCall2b
/batch_normalization_673/StatefulPartitionedCall/batch_normalization_673/StatefulPartitionedCall2F
!dense_734/StatefulPartitionedCall!dense_734/StatefulPartitionedCall2F
!dense_735/StatefulPartitionedCall!dense_735/StatefulPartitionedCall2F
!dense_736/StatefulPartitionedCall!dense_736/StatefulPartitionedCall2F
!dense_737/StatefulPartitionedCall!dense_737/StatefulPartitionedCall2F
!dense_738/StatefulPartitionedCall!dense_738/StatefulPartitionedCall2F
!dense_739/StatefulPartitionedCall!dense_739/StatefulPartitionedCall2F
!dense_740/StatefulPartitionedCall!dense_740/StatefulPartitionedCall2F
!dense_741/StatefulPartitionedCall!dense_741/StatefulPartitionedCall2F
!dense_742/StatefulPartitionedCall!dense_742/StatefulPartitionedCall2F
!dense_743/StatefulPartitionedCall!dense_743/StatefulPartitionedCall:O K
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
*__inference_dense_736_layer_call_fn_736356

inputs
unknown:Sa
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
E__inference_dense_736_layer_call_and_return_conditional_losses_733819o
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
:ÿÿÿÿÿÿÿÿÿS: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_671_layer_call_and_return_conditional_losses_733509

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
«
L
0__inference_leaky_re_lu_668_layer_call_fn_736560

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
K__inference_leaky_re_lu_668_layer_call_and_return_conditional_losses_733871`
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
¬
Ó
8__inference_batch_normalization_668_layer_call_fn_736488

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
S__inference_batch_normalization_668_layer_call_and_return_conditional_losses_733263o
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
8__inference_batch_normalization_670_layer_call_fn_736719

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_670_layer_call_and_return_conditional_losses_733474o
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
ª
Ó
8__inference_batch_normalization_668_layer_call_fn_736501

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
S__inference_batch_normalization_668_layer_call_and_return_conditional_losses_733310o
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
S__inference_batch_normalization_667_layer_call_and_return_conditional_losses_733228

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
È	
ö
E__inference_dense_734_layer_call_and_return_conditional_losses_736148

inputs0
matmul_readvariableop_resource:S-
biasadd_readvariableop_resource:S
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:S*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:S*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSw
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
S__inference_batch_normalization_669_layer_call_and_return_conditional_losses_733345

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
å
g
K__inference_leaky_re_lu_665_layer_call_and_return_conditional_losses_733775

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿS:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
Õ
û
I__inference_sequential_69_layer_call_and_return_conditional_losses_734988
normalization_69_input
normalization_69_sub_y
normalization_69_sqrt_x"
dense_734_734847:S
dense_734_734849:S,
batch_normalization_665_734852:S,
batch_normalization_665_734854:S,
batch_normalization_665_734856:S,
batch_normalization_665_734858:S"
dense_735_734862:SS
dense_735_734864:S,
batch_normalization_666_734867:S,
batch_normalization_666_734869:S,
batch_normalization_666_734871:S,
batch_normalization_666_734873:S"
dense_736_734877:Sa
dense_736_734879:a,
batch_normalization_667_734882:a,
batch_normalization_667_734884:a,
batch_normalization_667_734886:a,
batch_normalization_667_734888:a"
dense_737_734892:aa
dense_737_734894:a,
batch_normalization_668_734897:a,
batch_normalization_668_734899:a,
batch_normalization_668_734901:a,
batch_normalization_668_734903:a"
dense_738_734907:aA
dense_738_734909:A,
batch_normalization_669_734912:A,
batch_normalization_669_734914:A,
batch_normalization_669_734916:A,
batch_normalization_669_734918:A"
dense_739_734922:AA
dense_739_734924:A,
batch_normalization_670_734927:A,
batch_normalization_670_734929:A,
batch_normalization_670_734931:A,
batch_normalization_670_734933:A"
dense_740_734937:AA
dense_740_734939:A,
batch_normalization_671_734942:A,
batch_normalization_671_734944:A,
batch_normalization_671_734946:A,
batch_normalization_671_734948:A"
dense_741_734952:AA
dense_741_734954:A,
batch_normalization_672_734957:A,
batch_normalization_672_734959:A,
batch_normalization_672_734961:A,
batch_normalization_672_734963:A"
dense_742_734967:AA
dense_742_734969:A,
batch_normalization_673_734972:A,
batch_normalization_673_734974:A,
batch_normalization_673_734976:A,
batch_normalization_673_734978:A"
dense_743_734982:A
dense_743_734984:
identity¢/batch_normalization_665/StatefulPartitionedCall¢/batch_normalization_666/StatefulPartitionedCall¢/batch_normalization_667/StatefulPartitionedCall¢/batch_normalization_668/StatefulPartitionedCall¢/batch_normalization_669/StatefulPartitionedCall¢/batch_normalization_670/StatefulPartitionedCall¢/batch_normalization_671/StatefulPartitionedCall¢/batch_normalization_672/StatefulPartitionedCall¢/batch_normalization_673/StatefulPartitionedCall¢!dense_734/StatefulPartitionedCall¢!dense_735/StatefulPartitionedCall¢!dense_736/StatefulPartitionedCall¢!dense_737/StatefulPartitionedCall¢!dense_738/StatefulPartitionedCall¢!dense_739/StatefulPartitionedCall¢!dense_740/StatefulPartitionedCall¢!dense_741/StatefulPartitionedCall¢!dense_742/StatefulPartitionedCall¢!dense_743/StatefulPartitionedCall}
normalization_69/subSubnormalization_69_inputnormalization_69_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_69/SqrtSqrtnormalization_69_sqrt_x*
T0*
_output_shapes

:_
normalization_69/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_69/MaximumMaximumnormalization_69/Sqrt:y:0#normalization_69/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_69/truedivRealDivnormalization_69/sub:z:0normalization_69/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_734/StatefulPartitionedCallStatefulPartitionedCallnormalization_69/truediv:z:0dense_734_734847dense_734_734849*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_734_layer_call_and_return_conditional_losses_733755
/batch_normalization_665/StatefulPartitionedCallStatefulPartitionedCall*dense_734/StatefulPartitionedCall:output:0batch_normalization_665_734852batch_normalization_665_734854batch_normalization_665_734856batch_normalization_665_734858*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_665_layer_call_and_return_conditional_losses_733017ø
leaky_re_lu_665/PartitionedCallPartitionedCall8batch_normalization_665/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_665_layer_call_and_return_conditional_losses_733775
!dense_735/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_665/PartitionedCall:output:0dense_735_734862dense_735_734864*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_735_layer_call_and_return_conditional_losses_733787
/batch_normalization_666/StatefulPartitionedCallStatefulPartitionedCall*dense_735/StatefulPartitionedCall:output:0batch_normalization_666_734867batch_normalization_666_734869batch_normalization_666_734871batch_normalization_666_734873*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_666_layer_call_and_return_conditional_losses_733099ø
leaky_re_lu_666/PartitionedCallPartitionedCall8batch_normalization_666/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_666_layer_call_and_return_conditional_losses_733807
!dense_736/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_666/PartitionedCall:output:0dense_736_734877dense_736_734879*
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
E__inference_dense_736_layer_call_and_return_conditional_losses_733819
/batch_normalization_667/StatefulPartitionedCallStatefulPartitionedCall*dense_736/StatefulPartitionedCall:output:0batch_normalization_667_734882batch_normalization_667_734884batch_normalization_667_734886batch_normalization_667_734888*
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
S__inference_batch_normalization_667_layer_call_and_return_conditional_losses_733181ø
leaky_re_lu_667/PartitionedCallPartitionedCall8batch_normalization_667/StatefulPartitionedCall:output:0*
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
K__inference_leaky_re_lu_667_layer_call_and_return_conditional_losses_733839
!dense_737/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_667/PartitionedCall:output:0dense_737_734892dense_737_734894*
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
E__inference_dense_737_layer_call_and_return_conditional_losses_733851
/batch_normalization_668/StatefulPartitionedCallStatefulPartitionedCall*dense_737/StatefulPartitionedCall:output:0batch_normalization_668_734897batch_normalization_668_734899batch_normalization_668_734901batch_normalization_668_734903*
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
S__inference_batch_normalization_668_layer_call_and_return_conditional_losses_733263ø
leaky_re_lu_668/PartitionedCallPartitionedCall8batch_normalization_668/StatefulPartitionedCall:output:0*
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
K__inference_leaky_re_lu_668_layer_call_and_return_conditional_losses_733871
!dense_738/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_668/PartitionedCall:output:0dense_738_734907dense_738_734909*
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
GPU 2J 8 *N
fIRG
E__inference_dense_738_layer_call_and_return_conditional_losses_733883
/batch_normalization_669/StatefulPartitionedCallStatefulPartitionedCall*dense_738/StatefulPartitionedCall:output:0batch_normalization_669_734912batch_normalization_669_734914batch_normalization_669_734916batch_normalization_669_734918*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_669_layer_call_and_return_conditional_losses_733345ø
leaky_re_lu_669/PartitionedCallPartitionedCall8batch_normalization_669/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_669_layer_call_and_return_conditional_losses_733903
!dense_739/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_669/PartitionedCall:output:0dense_739_734922dense_739_734924*
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
GPU 2J 8 *N
fIRG
E__inference_dense_739_layer_call_and_return_conditional_losses_733915
/batch_normalization_670/StatefulPartitionedCallStatefulPartitionedCall*dense_739/StatefulPartitionedCall:output:0batch_normalization_670_734927batch_normalization_670_734929batch_normalization_670_734931batch_normalization_670_734933*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_670_layer_call_and_return_conditional_losses_733427ø
leaky_re_lu_670/PartitionedCallPartitionedCall8batch_normalization_670/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_670_layer_call_and_return_conditional_losses_733935
!dense_740/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_670/PartitionedCall:output:0dense_740_734937dense_740_734939*
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
GPU 2J 8 *N
fIRG
E__inference_dense_740_layer_call_and_return_conditional_losses_733947
/batch_normalization_671/StatefulPartitionedCallStatefulPartitionedCall*dense_740/StatefulPartitionedCall:output:0batch_normalization_671_734942batch_normalization_671_734944batch_normalization_671_734946batch_normalization_671_734948*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_671_layer_call_and_return_conditional_losses_733509ø
leaky_re_lu_671/PartitionedCallPartitionedCall8batch_normalization_671/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_671_layer_call_and_return_conditional_losses_733967
!dense_741/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_671/PartitionedCall:output:0dense_741_734952dense_741_734954*
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
GPU 2J 8 *N
fIRG
E__inference_dense_741_layer_call_and_return_conditional_losses_733979
/batch_normalization_672/StatefulPartitionedCallStatefulPartitionedCall*dense_741/StatefulPartitionedCall:output:0batch_normalization_672_734957batch_normalization_672_734959batch_normalization_672_734961batch_normalization_672_734963*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_672_layer_call_and_return_conditional_losses_733591ø
leaky_re_lu_672/PartitionedCallPartitionedCall8batch_normalization_672/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_672_layer_call_and_return_conditional_losses_733999
!dense_742/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_672/PartitionedCall:output:0dense_742_734967dense_742_734969*
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
GPU 2J 8 *N
fIRG
E__inference_dense_742_layer_call_and_return_conditional_losses_734011
/batch_normalization_673/StatefulPartitionedCallStatefulPartitionedCall*dense_742/StatefulPartitionedCall:output:0batch_normalization_673_734972batch_normalization_673_734974batch_normalization_673_734976batch_normalization_673_734978*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_673_layer_call_and_return_conditional_losses_733673ø
leaky_re_lu_673/PartitionedCallPartitionedCall8batch_normalization_673/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_673_layer_call_and_return_conditional_losses_734031
!dense_743/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_673/PartitionedCall:output:0dense_743_734982dense_743_734984*
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
E__inference_dense_743_layer_call_and_return_conditional_losses_734043y
IdentityIdentity*dense_743/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp0^batch_normalization_665/StatefulPartitionedCall0^batch_normalization_666/StatefulPartitionedCall0^batch_normalization_667/StatefulPartitionedCall0^batch_normalization_668/StatefulPartitionedCall0^batch_normalization_669/StatefulPartitionedCall0^batch_normalization_670/StatefulPartitionedCall0^batch_normalization_671/StatefulPartitionedCall0^batch_normalization_672/StatefulPartitionedCall0^batch_normalization_673/StatefulPartitionedCall"^dense_734/StatefulPartitionedCall"^dense_735/StatefulPartitionedCall"^dense_736/StatefulPartitionedCall"^dense_737/StatefulPartitionedCall"^dense_738/StatefulPartitionedCall"^dense_739/StatefulPartitionedCall"^dense_740/StatefulPartitionedCall"^dense_741/StatefulPartitionedCall"^dense_742/StatefulPartitionedCall"^dense_743/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_665/StatefulPartitionedCall/batch_normalization_665/StatefulPartitionedCall2b
/batch_normalization_666/StatefulPartitionedCall/batch_normalization_666/StatefulPartitionedCall2b
/batch_normalization_667/StatefulPartitionedCall/batch_normalization_667/StatefulPartitionedCall2b
/batch_normalization_668/StatefulPartitionedCall/batch_normalization_668/StatefulPartitionedCall2b
/batch_normalization_669/StatefulPartitionedCall/batch_normalization_669/StatefulPartitionedCall2b
/batch_normalization_670/StatefulPartitionedCall/batch_normalization_670/StatefulPartitionedCall2b
/batch_normalization_671/StatefulPartitionedCall/batch_normalization_671/StatefulPartitionedCall2b
/batch_normalization_672/StatefulPartitionedCall/batch_normalization_672/StatefulPartitionedCall2b
/batch_normalization_673/StatefulPartitionedCall/batch_normalization_673/StatefulPartitionedCall2F
!dense_734/StatefulPartitionedCall!dense_734/StatefulPartitionedCall2F
!dense_735/StatefulPartitionedCall!dense_735/StatefulPartitionedCall2F
!dense_736/StatefulPartitionedCall!dense_736/StatefulPartitionedCall2F
!dense_737/StatefulPartitionedCall!dense_737/StatefulPartitionedCall2F
!dense_738/StatefulPartitionedCall!dense_738/StatefulPartitionedCall2F
!dense_739/StatefulPartitionedCall!dense_739/StatefulPartitionedCall2F
!dense_740/StatefulPartitionedCall!dense_740/StatefulPartitionedCall2F
!dense_741/StatefulPartitionedCall!dense_741/StatefulPartitionedCall2F
!dense_742/StatefulPartitionedCall!dense_742/StatefulPartitionedCall2F
!dense_743/StatefulPartitionedCall!dense_743/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_69_input:$ 

_output_shapes

::$ 

_output_shapes

:
ª
Ó
8__inference_batch_normalization_665_layer_call_fn_736174

inputs
unknown:S
	unknown_0:S
	unknown_1:S
	unknown_2:S
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_665_layer_call_and_return_conditional_losses_733064o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿS: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
È	
ö
E__inference_dense_735_layer_call_and_return_conditional_losses_733787

inputs0
matmul_readvariableop_resource:SS-
biasadd_readvariableop_resource:S
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:SS*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:S*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿS: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
 ¯
;
I__inference_sequential_69_layer_call_and_return_conditional_losses_735959

inputs
normalization_69_sub_y
normalization_69_sqrt_x:
(dense_734_matmul_readvariableop_resource:S7
)dense_734_biasadd_readvariableop_resource:SM
?batch_normalization_665_assignmovingavg_readvariableop_resource:SO
Abatch_normalization_665_assignmovingavg_1_readvariableop_resource:SK
=batch_normalization_665_batchnorm_mul_readvariableop_resource:SG
9batch_normalization_665_batchnorm_readvariableop_resource:S:
(dense_735_matmul_readvariableop_resource:SS7
)dense_735_biasadd_readvariableop_resource:SM
?batch_normalization_666_assignmovingavg_readvariableop_resource:SO
Abatch_normalization_666_assignmovingavg_1_readvariableop_resource:SK
=batch_normalization_666_batchnorm_mul_readvariableop_resource:SG
9batch_normalization_666_batchnorm_readvariableop_resource:S:
(dense_736_matmul_readvariableop_resource:Sa7
)dense_736_biasadd_readvariableop_resource:aM
?batch_normalization_667_assignmovingavg_readvariableop_resource:aO
Abatch_normalization_667_assignmovingavg_1_readvariableop_resource:aK
=batch_normalization_667_batchnorm_mul_readvariableop_resource:aG
9batch_normalization_667_batchnorm_readvariableop_resource:a:
(dense_737_matmul_readvariableop_resource:aa7
)dense_737_biasadd_readvariableop_resource:aM
?batch_normalization_668_assignmovingavg_readvariableop_resource:aO
Abatch_normalization_668_assignmovingavg_1_readvariableop_resource:aK
=batch_normalization_668_batchnorm_mul_readvariableop_resource:aG
9batch_normalization_668_batchnorm_readvariableop_resource:a:
(dense_738_matmul_readvariableop_resource:aA7
)dense_738_biasadd_readvariableop_resource:AM
?batch_normalization_669_assignmovingavg_readvariableop_resource:AO
Abatch_normalization_669_assignmovingavg_1_readvariableop_resource:AK
=batch_normalization_669_batchnorm_mul_readvariableop_resource:AG
9batch_normalization_669_batchnorm_readvariableop_resource:A:
(dense_739_matmul_readvariableop_resource:AA7
)dense_739_biasadd_readvariableop_resource:AM
?batch_normalization_670_assignmovingavg_readvariableop_resource:AO
Abatch_normalization_670_assignmovingavg_1_readvariableop_resource:AK
=batch_normalization_670_batchnorm_mul_readvariableop_resource:AG
9batch_normalization_670_batchnorm_readvariableop_resource:A:
(dense_740_matmul_readvariableop_resource:AA7
)dense_740_biasadd_readvariableop_resource:AM
?batch_normalization_671_assignmovingavg_readvariableop_resource:AO
Abatch_normalization_671_assignmovingavg_1_readvariableop_resource:AK
=batch_normalization_671_batchnorm_mul_readvariableop_resource:AG
9batch_normalization_671_batchnorm_readvariableop_resource:A:
(dense_741_matmul_readvariableop_resource:AA7
)dense_741_biasadd_readvariableop_resource:AM
?batch_normalization_672_assignmovingavg_readvariableop_resource:AO
Abatch_normalization_672_assignmovingavg_1_readvariableop_resource:AK
=batch_normalization_672_batchnorm_mul_readvariableop_resource:AG
9batch_normalization_672_batchnorm_readvariableop_resource:A:
(dense_742_matmul_readvariableop_resource:AA7
)dense_742_biasadd_readvariableop_resource:AM
?batch_normalization_673_assignmovingavg_readvariableop_resource:AO
Abatch_normalization_673_assignmovingavg_1_readvariableop_resource:AK
=batch_normalization_673_batchnorm_mul_readvariableop_resource:AG
9batch_normalization_673_batchnorm_readvariableop_resource:A:
(dense_743_matmul_readvariableop_resource:A7
)dense_743_biasadd_readvariableop_resource:
identity¢'batch_normalization_665/AssignMovingAvg¢6batch_normalization_665/AssignMovingAvg/ReadVariableOp¢)batch_normalization_665/AssignMovingAvg_1¢8batch_normalization_665/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_665/batchnorm/ReadVariableOp¢4batch_normalization_665/batchnorm/mul/ReadVariableOp¢'batch_normalization_666/AssignMovingAvg¢6batch_normalization_666/AssignMovingAvg/ReadVariableOp¢)batch_normalization_666/AssignMovingAvg_1¢8batch_normalization_666/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_666/batchnorm/ReadVariableOp¢4batch_normalization_666/batchnorm/mul/ReadVariableOp¢'batch_normalization_667/AssignMovingAvg¢6batch_normalization_667/AssignMovingAvg/ReadVariableOp¢)batch_normalization_667/AssignMovingAvg_1¢8batch_normalization_667/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_667/batchnorm/ReadVariableOp¢4batch_normalization_667/batchnorm/mul/ReadVariableOp¢'batch_normalization_668/AssignMovingAvg¢6batch_normalization_668/AssignMovingAvg/ReadVariableOp¢)batch_normalization_668/AssignMovingAvg_1¢8batch_normalization_668/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_668/batchnorm/ReadVariableOp¢4batch_normalization_668/batchnorm/mul/ReadVariableOp¢'batch_normalization_669/AssignMovingAvg¢6batch_normalization_669/AssignMovingAvg/ReadVariableOp¢)batch_normalization_669/AssignMovingAvg_1¢8batch_normalization_669/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_669/batchnorm/ReadVariableOp¢4batch_normalization_669/batchnorm/mul/ReadVariableOp¢'batch_normalization_670/AssignMovingAvg¢6batch_normalization_670/AssignMovingAvg/ReadVariableOp¢)batch_normalization_670/AssignMovingAvg_1¢8batch_normalization_670/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_670/batchnorm/ReadVariableOp¢4batch_normalization_670/batchnorm/mul/ReadVariableOp¢'batch_normalization_671/AssignMovingAvg¢6batch_normalization_671/AssignMovingAvg/ReadVariableOp¢)batch_normalization_671/AssignMovingAvg_1¢8batch_normalization_671/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_671/batchnorm/ReadVariableOp¢4batch_normalization_671/batchnorm/mul/ReadVariableOp¢'batch_normalization_672/AssignMovingAvg¢6batch_normalization_672/AssignMovingAvg/ReadVariableOp¢)batch_normalization_672/AssignMovingAvg_1¢8batch_normalization_672/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_672/batchnorm/ReadVariableOp¢4batch_normalization_672/batchnorm/mul/ReadVariableOp¢'batch_normalization_673/AssignMovingAvg¢6batch_normalization_673/AssignMovingAvg/ReadVariableOp¢)batch_normalization_673/AssignMovingAvg_1¢8batch_normalization_673/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_673/batchnorm/ReadVariableOp¢4batch_normalization_673/batchnorm/mul/ReadVariableOp¢ dense_734/BiasAdd/ReadVariableOp¢dense_734/MatMul/ReadVariableOp¢ dense_735/BiasAdd/ReadVariableOp¢dense_735/MatMul/ReadVariableOp¢ dense_736/BiasAdd/ReadVariableOp¢dense_736/MatMul/ReadVariableOp¢ dense_737/BiasAdd/ReadVariableOp¢dense_737/MatMul/ReadVariableOp¢ dense_738/BiasAdd/ReadVariableOp¢dense_738/MatMul/ReadVariableOp¢ dense_739/BiasAdd/ReadVariableOp¢dense_739/MatMul/ReadVariableOp¢ dense_740/BiasAdd/ReadVariableOp¢dense_740/MatMul/ReadVariableOp¢ dense_741/BiasAdd/ReadVariableOp¢dense_741/MatMul/ReadVariableOp¢ dense_742/BiasAdd/ReadVariableOp¢dense_742/MatMul/ReadVariableOp¢ dense_743/BiasAdd/ReadVariableOp¢dense_743/MatMul/ReadVariableOpm
normalization_69/subSubinputsnormalization_69_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_69/SqrtSqrtnormalization_69_sqrt_x*
T0*
_output_shapes

:_
normalization_69/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_69/MaximumMaximumnormalization_69/Sqrt:y:0#normalization_69/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_69/truedivRealDivnormalization_69/sub:z:0normalization_69/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_734/MatMul/ReadVariableOpReadVariableOp(dense_734_matmul_readvariableop_resource*
_output_shapes

:S*
dtype0
dense_734/MatMulMatMulnormalization_69/truediv:z:0'dense_734/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 dense_734/BiasAdd/ReadVariableOpReadVariableOp)dense_734_biasadd_readvariableop_resource*
_output_shapes
:S*
dtype0
dense_734/BiasAddBiasAdddense_734/MatMul:product:0(dense_734/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
6batch_normalization_665/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_665/moments/meanMeandense_734/BiasAdd:output:0?batch_normalization_665/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:S*
	keep_dims(
,batch_normalization_665/moments/StopGradientStopGradient-batch_normalization_665/moments/mean:output:0*
T0*
_output_shapes

:SË
1batch_normalization_665/moments/SquaredDifferenceSquaredDifferencedense_734/BiasAdd:output:05batch_normalization_665/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
:batch_normalization_665/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_665/moments/varianceMean5batch_normalization_665/moments/SquaredDifference:z:0Cbatch_normalization_665/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:S*
	keep_dims(
'batch_normalization_665/moments/SqueezeSqueeze-batch_normalization_665/moments/mean:output:0*
T0*
_output_shapes
:S*
squeeze_dims
 £
)batch_normalization_665/moments/Squeeze_1Squeeze1batch_normalization_665/moments/variance:output:0*
T0*
_output_shapes
:S*
squeeze_dims
 r
-batch_normalization_665/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_665/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_665_assignmovingavg_readvariableop_resource*
_output_shapes
:S*
dtype0É
+batch_normalization_665/AssignMovingAvg/subSub>batch_normalization_665/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_665/moments/Squeeze:output:0*
T0*
_output_shapes
:SÀ
+batch_normalization_665/AssignMovingAvg/mulMul/batch_normalization_665/AssignMovingAvg/sub:z:06batch_normalization_665/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:S
'batch_normalization_665/AssignMovingAvgAssignSubVariableOp?batch_normalization_665_assignmovingavg_readvariableop_resource/batch_normalization_665/AssignMovingAvg/mul:z:07^batch_normalization_665/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_665/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_665/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_665_assignmovingavg_1_readvariableop_resource*
_output_shapes
:S*
dtype0Ï
-batch_normalization_665/AssignMovingAvg_1/subSub@batch_normalization_665/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_665/moments/Squeeze_1:output:0*
T0*
_output_shapes
:SÆ
-batch_normalization_665/AssignMovingAvg_1/mulMul1batch_normalization_665/AssignMovingAvg_1/sub:z:08batch_normalization_665/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:S
)batch_normalization_665/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_665_assignmovingavg_1_readvariableop_resource1batch_normalization_665/AssignMovingAvg_1/mul:z:09^batch_normalization_665/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_665/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_665/batchnorm/addAddV22batch_normalization_665/moments/Squeeze_1:output:00batch_normalization_665/batchnorm/add/y:output:0*
T0*
_output_shapes
:S
'batch_normalization_665/batchnorm/RsqrtRsqrt)batch_normalization_665/batchnorm/add:z:0*
T0*
_output_shapes
:S®
4batch_normalization_665/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_665_batchnorm_mul_readvariableop_resource*
_output_shapes
:S*
dtype0¼
%batch_normalization_665/batchnorm/mulMul+batch_normalization_665/batchnorm/Rsqrt:y:0<batch_normalization_665/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:S§
'batch_normalization_665/batchnorm/mul_1Muldense_734/BiasAdd:output:0)batch_normalization_665/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS°
'batch_normalization_665/batchnorm/mul_2Mul0batch_normalization_665/moments/Squeeze:output:0)batch_normalization_665/batchnorm/mul:z:0*
T0*
_output_shapes
:S¦
0batch_normalization_665/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_665_batchnorm_readvariableop_resource*
_output_shapes
:S*
dtype0¸
%batch_normalization_665/batchnorm/subSub8batch_normalization_665/batchnorm/ReadVariableOp:value:0+batch_normalization_665/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Sº
'batch_normalization_665/batchnorm/add_1AddV2+batch_normalization_665/batchnorm/mul_1:z:0)batch_normalization_665/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
leaky_re_lu_665/LeakyRelu	LeakyRelu+batch_normalization_665/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
alpha%>
dense_735/MatMul/ReadVariableOpReadVariableOp(dense_735_matmul_readvariableop_resource*
_output_shapes

:SS*
dtype0
dense_735/MatMulMatMul'leaky_re_lu_665/LeakyRelu:activations:0'dense_735/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 dense_735/BiasAdd/ReadVariableOpReadVariableOp)dense_735_biasadd_readvariableop_resource*
_output_shapes
:S*
dtype0
dense_735/BiasAddBiasAdddense_735/MatMul:product:0(dense_735/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
6batch_normalization_666/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_666/moments/meanMeandense_735/BiasAdd:output:0?batch_normalization_666/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:S*
	keep_dims(
,batch_normalization_666/moments/StopGradientStopGradient-batch_normalization_666/moments/mean:output:0*
T0*
_output_shapes

:SË
1batch_normalization_666/moments/SquaredDifferenceSquaredDifferencedense_735/BiasAdd:output:05batch_normalization_666/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
:batch_normalization_666/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_666/moments/varianceMean5batch_normalization_666/moments/SquaredDifference:z:0Cbatch_normalization_666/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:S*
	keep_dims(
'batch_normalization_666/moments/SqueezeSqueeze-batch_normalization_666/moments/mean:output:0*
T0*
_output_shapes
:S*
squeeze_dims
 £
)batch_normalization_666/moments/Squeeze_1Squeeze1batch_normalization_666/moments/variance:output:0*
T0*
_output_shapes
:S*
squeeze_dims
 r
-batch_normalization_666/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_666/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_666_assignmovingavg_readvariableop_resource*
_output_shapes
:S*
dtype0É
+batch_normalization_666/AssignMovingAvg/subSub>batch_normalization_666/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_666/moments/Squeeze:output:0*
T0*
_output_shapes
:SÀ
+batch_normalization_666/AssignMovingAvg/mulMul/batch_normalization_666/AssignMovingAvg/sub:z:06batch_normalization_666/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:S
'batch_normalization_666/AssignMovingAvgAssignSubVariableOp?batch_normalization_666_assignmovingavg_readvariableop_resource/batch_normalization_666/AssignMovingAvg/mul:z:07^batch_normalization_666/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_666/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_666/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_666_assignmovingavg_1_readvariableop_resource*
_output_shapes
:S*
dtype0Ï
-batch_normalization_666/AssignMovingAvg_1/subSub@batch_normalization_666/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_666/moments/Squeeze_1:output:0*
T0*
_output_shapes
:SÆ
-batch_normalization_666/AssignMovingAvg_1/mulMul1batch_normalization_666/AssignMovingAvg_1/sub:z:08batch_normalization_666/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:S
)batch_normalization_666/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_666_assignmovingavg_1_readvariableop_resource1batch_normalization_666/AssignMovingAvg_1/mul:z:09^batch_normalization_666/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_666/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_666/batchnorm/addAddV22batch_normalization_666/moments/Squeeze_1:output:00batch_normalization_666/batchnorm/add/y:output:0*
T0*
_output_shapes
:S
'batch_normalization_666/batchnorm/RsqrtRsqrt)batch_normalization_666/batchnorm/add:z:0*
T0*
_output_shapes
:S®
4batch_normalization_666/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_666_batchnorm_mul_readvariableop_resource*
_output_shapes
:S*
dtype0¼
%batch_normalization_666/batchnorm/mulMul+batch_normalization_666/batchnorm/Rsqrt:y:0<batch_normalization_666/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:S§
'batch_normalization_666/batchnorm/mul_1Muldense_735/BiasAdd:output:0)batch_normalization_666/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS°
'batch_normalization_666/batchnorm/mul_2Mul0batch_normalization_666/moments/Squeeze:output:0)batch_normalization_666/batchnorm/mul:z:0*
T0*
_output_shapes
:S¦
0batch_normalization_666/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_666_batchnorm_readvariableop_resource*
_output_shapes
:S*
dtype0¸
%batch_normalization_666/batchnorm/subSub8batch_normalization_666/batchnorm/ReadVariableOp:value:0+batch_normalization_666/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Sº
'batch_normalization_666/batchnorm/add_1AddV2+batch_normalization_666/batchnorm/mul_1:z:0)batch_normalization_666/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
leaky_re_lu_666/LeakyRelu	LeakyRelu+batch_normalization_666/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
alpha%>
dense_736/MatMul/ReadVariableOpReadVariableOp(dense_736_matmul_readvariableop_resource*
_output_shapes

:Sa*
dtype0
dense_736/MatMulMatMul'leaky_re_lu_666/LeakyRelu:activations:0'dense_736/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 dense_736/BiasAdd/ReadVariableOpReadVariableOp)dense_736_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
dense_736/BiasAddBiasAdddense_736/MatMul:product:0(dense_736/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
6batch_normalization_667/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_667/moments/meanMeandense_736/BiasAdd:output:0?batch_normalization_667/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(
,batch_normalization_667/moments/StopGradientStopGradient-batch_normalization_667/moments/mean:output:0*
T0*
_output_shapes

:aË
1batch_normalization_667/moments/SquaredDifferenceSquaredDifferencedense_736/BiasAdd:output:05batch_normalization_667/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
:batch_normalization_667/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_667/moments/varianceMean5batch_normalization_667/moments/SquaredDifference:z:0Cbatch_normalization_667/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(
'batch_normalization_667/moments/SqueezeSqueeze-batch_normalization_667/moments/mean:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 £
)batch_normalization_667/moments/Squeeze_1Squeeze1batch_normalization_667/moments/variance:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 r
-batch_normalization_667/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_667/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_667_assignmovingavg_readvariableop_resource*
_output_shapes
:a*
dtype0É
+batch_normalization_667/AssignMovingAvg/subSub>batch_normalization_667/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_667/moments/Squeeze:output:0*
T0*
_output_shapes
:aÀ
+batch_normalization_667/AssignMovingAvg/mulMul/batch_normalization_667/AssignMovingAvg/sub:z:06batch_normalization_667/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:a
'batch_normalization_667/AssignMovingAvgAssignSubVariableOp?batch_normalization_667_assignmovingavg_readvariableop_resource/batch_normalization_667/AssignMovingAvg/mul:z:07^batch_normalization_667/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_667/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_667/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_667_assignmovingavg_1_readvariableop_resource*
_output_shapes
:a*
dtype0Ï
-batch_normalization_667/AssignMovingAvg_1/subSub@batch_normalization_667/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_667/moments/Squeeze_1:output:0*
T0*
_output_shapes
:aÆ
-batch_normalization_667/AssignMovingAvg_1/mulMul1batch_normalization_667/AssignMovingAvg_1/sub:z:08batch_normalization_667/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:a
)batch_normalization_667/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_667_assignmovingavg_1_readvariableop_resource1batch_normalization_667/AssignMovingAvg_1/mul:z:09^batch_normalization_667/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_667/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_667/batchnorm/addAddV22batch_normalization_667/moments/Squeeze_1:output:00batch_normalization_667/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
'batch_normalization_667/batchnorm/RsqrtRsqrt)batch_normalization_667/batchnorm/add:z:0*
T0*
_output_shapes
:a®
4batch_normalization_667/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_667_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0¼
%batch_normalization_667/batchnorm/mulMul+batch_normalization_667/batchnorm/Rsqrt:y:0<batch_normalization_667/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:a§
'batch_normalization_667/batchnorm/mul_1Muldense_736/BiasAdd:output:0)batch_normalization_667/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa°
'batch_normalization_667/batchnorm/mul_2Mul0batch_normalization_667/moments/Squeeze:output:0)batch_normalization_667/batchnorm/mul:z:0*
T0*
_output_shapes
:a¦
0batch_normalization_667/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_667_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0¸
%batch_normalization_667/batchnorm/subSub8batch_normalization_667/batchnorm/ReadVariableOp:value:0+batch_normalization_667/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aº
'batch_normalization_667/batchnorm/add_1AddV2+batch_normalization_667/batchnorm/mul_1:z:0)batch_normalization_667/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
leaky_re_lu_667/LeakyRelu	LeakyRelu+batch_normalization_667/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>
dense_737/MatMul/ReadVariableOpReadVariableOp(dense_737_matmul_readvariableop_resource*
_output_shapes

:aa*
dtype0
dense_737/MatMulMatMul'leaky_re_lu_667/LeakyRelu:activations:0'dense_737/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 dense_737/BiasAdd/ReadVariableOpReadVariableOp)dense_737_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
dense_737/BiasAddBiasAdddense_737/MatMul:product:0(dense_737/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
6batch_normalization_668/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_668/moments/meanMeandense_737/BiasAdd:output:0?batch_normalization_668/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(
,batch_normalization_668/moments/StopGradientStopGradient-batch_normalization_668/moments/mean:output:0*
T0*
_output_shapes

:aË
1batch_normalization_668/moments/SquaredDifferenceSquaredDifferencedense_737/BiasAdd:output:05batch_normalization_668/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
:batch_normalization_668/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_668/moments/varianceMean5batch_normalization_668/moments/SquaredDifference:z:0Cbatch_normalization_668/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:a*
	keep_dims(
'batch_normalization_668/moments/SqueezeSqueeze-batch_normalization_668/moments/mean:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 £
)batch_normalization_668/moments/Squeeze_1Squeeze1batch_normalization_668/moments/variance:output:0*
T0*
_output_shapes
:a*
squeeze_dims
 r
-batch_normalization_668/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_668/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_668_assignmovingavg_readvariableop_resource*
_output_shapes
:a*
dtype0É
+batch_normalization_668/AssignMovingAvg/subSub>batch_normalization_668/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_668/moments/Squeeze:output:0*
T0*
_output_shapes
:aÀ
+batch_normalization_668/AssignMovingAvg/mulMul/batch_normalization_668/AssignMovingAvg/sub:z:06batch_normalization_668/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:a
'batch_normalization_668/AssignMovingAvgAssignSubVariableOp?batch_normalization_668_assignmovingavg_readvariableop_resource/batch_normalization_668/AssignMovingAvg/mul:z:07^batch_normalization_668/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_668/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_668/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_668_assignmovingavg_1_readvariableop_resource*
_output_shapes
:a*
dtype0Ï
-batch_normalization_668/AssignMovingAvg_1/subSub@batch_normalization_668/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_668/moments/Squeeze_1:output:0*
T0*
_output_shapes
:aÆ
-batch_normalization_668/AssignMovingAvg_1/mulMul1batch_normalization_668/AssignMovingAvg_1/sub:z:08batch_normalization_668/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:a
)batch_normalization_668/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_668_assignmovingavg_1_readvariableop_resource1batch_normalization_668/AssignMovingAvg_1/mul:z:09^batch_normalization_668/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_668/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_668/batchnorm/addAddV22batch_normalization_668/moments/Squeeze_1:output:00batch_normalization_668/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
'batch_normalization_668/batchnorm/RsqrtRsqrt)batch_normalization_668/batchnorm/add:z:0*
T0*
_output_shapes
:a®
4batch_normalization_668/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_668_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0¼
%batch_normalization_668/batchnorm/mulMul+batch_normalization_668/batchnorm/Rsqrt:y:0<batch_normalization_668/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:a§
'batch_normalization_668/batchnorm/mul_1Muldense_737/BiasAdd:output:0)batch_normalization_668/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa°
'batch_normalization_668/batchnorm/mul_2Mul0batch_normalization_668/moments/Squeeze:output:0)batch_normalization_668/batchnorm/mul:z:0*
T0*
_output_shapes
:a¦
0batch_normalization_668/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_668_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0¸
%batch_normalization_668/batchnorm/subSub8batch_normalization_668/batchnorm/ReadVariableOp:value:0+batch_normalization_668/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aº
'batch_normalization_668/batchnorm/add_1AddV2+batch_normalization_668/batchnorm/mul_1:z:0)batch_normalization_668/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
leaky_re_lu_668/LeakyRelu	LeakyRelu+batch_normalization_668/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>
dense_738/MatMul/ReadVariableOpReadVariableOp(dense_738_matmul_readvariableop_resource*
_output_shapes

:aA*
dtype0
dense_738/MatMulMatMul'leaky_re_lu_668/LeakyRelu:activations:0'dense_738/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_738/BiasAdd/ReadVariableOpReadVariableOp)dense_738_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_738/BiasAddBiasAdddense_738/MatMul:product:0(dense_738/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
6batch_normalization_669/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_669/moments/meanMeandense_738/BiasAdd:output:0?batch_normalization_669/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
,batch_normalization_669/moments/StopGradientStopGradient-batch_normalization_669/moments/mean:output:0*
T0*
_output_shapes

:AË
1batch_normalization_669/moments/SquaredDifferenceSquaredDifferencedense_738/BiasAdd:output:05batch_normalization_669/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
:batch_normalization_669/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_669/moments/varianceMean5batch_normalization_669/moments/SquaredDifference:z:0Cbatch_normalization_669/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
'batch_normalization_669/moments/SqueezeSqueeze-batch_normalization_669/moments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 £
)batch_normalization_669/moments/Squeeze_1Squeeze1batch_normalization_669/moments/variance:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 r
-batch_normalization_669/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_669/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_669_assignmovingavg_readvariableop_resource*
_output_shapes
:A*
dtype0É
+batch_normalization_669/AssignMovingAvg/subSub>batch_normalization_669/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_669/moments/Squeeze:output:0*
T0*
_output_shapes
:AÀ
+batch_normalization_669/AssignMovingAvg/mulMul/batch_normalization_669/AssignMovingAvg/sub:z:06batch_normalization_669/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A
'batch_normalization_669/AssignMovingAvgAssignSubVariableOp?batch_normalization_669_assignmovingavg_readvariableop_resource/batch_normalization_669/AssignMovingAvg/mul:z:07^batch_normalization_669/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_669/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_669/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_669_assignmovingavg_1_readvariableop_resource*
_output_shapes
:A*
dtype0Ï
-batch_normalization_669/AssignMovingAvg_1/subSub@batch_normalization_669/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_669/moments/Squeeze_1:output:0*
T0*
_output_shapes
:AÆ
-batch_normalization_669/AssignMovingAvg_1/mulMul1batch_normalization_669/AssignMovingAvg_1/sub:z:08batch_normalization_669/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A
)batch_normalization_669/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_669_assignmovingavg_1_readvariableop_resource1batch_normalization_669/AssignMovingAvg_1/mul:z:09^batch_normalization_669/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_669/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_669/batchnorm/addAddV22batch_normalization_669/moments/Squeeze_1:output:00batch_normalization_669/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
'batch_normalization_669/batchnorm/RsqrtRsqrt)batch_normalization_669/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_669/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_669_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_669/batchnorm/mulMul+batch_normalization_669/batchnorm/Rsqrt:y:0<batch_normalization_669/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_669/batchnorm/mul_1Muldense_738/BiasAdd:output:0)batch_normalization_669/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA°
'batch_normalization_669/batchnorm/mul_2Mul0batch_normalization_669/moments/Squeeze:output:0)batch_normalization_669/batchnorm/mul:z:0*
T0*
_output_shapes
:A¦
0batch_normalization_669/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_669_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0¸
%batch_normalization_669/batchnorm/subSub8batch_normalization_669/batchnorm/ReadVariableOp:value:0+batch_normalization_669/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_669/batchnorm/add_1AddV2+batch_normalization_669/batchnorm/mul_1:z:0)batch_normalization_669/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_669/LeakyRelu	LeakyRelu+batch_normalization_669/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_739/MatMul/ReadVariableOpReadVariableOp(dense_739_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0
dense_739/MatMulMatMul'leaky_re_lu_669/LeakyRelu:activations:0'dense_739/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_739/BiasAdd/ReadVariableOpReadVariableOp)dense_739_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_739/BiasAddBiasAdddense_739/MatMul:product:0(dense_739/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
6batch_normalization_670/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_670/moments/meanMeandense_739/BiasAdd:output:0?batch_normalization_670/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
,batch_normalization_670/moments/StopGradientStopGradient-batch_normalization_670/moments/mean:output:0*
T0*
_output_shapes

:AË
1batch_normalization_670/moments/SquaredDifferenceSquaredDifferencedense_739/BiasAdd:output:05batch_normalization_670/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
:batch_normalization_670/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_670/moments/varianceMean5batch_normalization_670/moments/SquaredDifference:z:0Cbatch_normalization_670/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
'batch_normalization_670/moments/SqueezeSqueeze-batch_normalization_670/moments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 £
)batch_normalization_670/moments/Squeeze_1Squeeze1batch_normalization_670/moments/variance:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 r
-batch_normalization_670/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_670/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_670_assignmovingavg_readvariableop_resource*
_output_shapes
:A*
dtype0É
+batch_normalization_670/AssignMovingAvg/subSub>batch_normalization_670/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_670/moments/Squeeze:output:0*
T0*
_output_shapes
:AÀ
+batch_normalization_670/AssignMovingAvg/mulMul/batch_normalization_670/AssignMovingAvg/sub:z:06batch_normalization_670/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A
'batch_normalization_670/AssignMovingAvgAssignSubVariableOp?batch_normalization_670_assignmovingavg_readvariableop_resource/batch_normalization_670/AssignMovingAvg/mul:z:07^batch_normalization_670/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_670/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_670/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_670_assignmovingavg_1_readvariableop_resource*
_output_shapes
:A*
dtype0Ï
-batch_normalization_670/AssignMovingAvg_1/subSub@batch_normalization_670/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_670/moments/Squeeze_1:output:0*
T0*
_output_shapes
:AÆ
-batch_normalization_670/AssignMovingAvg_1/mulMul1batch_normalization_670/AssignMovingAvg_1/sub:z:08batch_normalization_670/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A
)batch_normalization_670/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_670_assignmovingavg_1_readvariableop_resource1batch_normalization_670/AssignMovingAvg_1/mul:z:09^batch_normalization_670/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_670/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_670/batchnorm/addAddV22batch_normalization_670/moments/Squeeze_1:output:00batch_normalization_670/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
'batch_normalization_670/batchnorm/RsqrtRsqrt)batch_normalization_670/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_670/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_670_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_670/batchnorm/mulMul+batch_normalization_670/batchnorm/Rsqrt:y:0<batch_normalization_670/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_670/batchnorm/mul_1Muldense_739/BiasAdd:output:0)batch_normalization_670/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA°
'batch_normalization_670/batchnorm/mul_2Mul0batch_normalization_670/moments/Squeeze:output:0)batch_normalization_670/batchnorm/mul:z:0*
T0*
_output_shapes
:A¦
0batch_normalization_670/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_670_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0¸
%batch_normalization_670/batchnorm/subSub8batch_normalization_670/batchnorm/ReadVariableOp:value:0+batch_normalization_670/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_670/batchnorm/add_1AddV2+batch_normalization_670/batchnorm/mul_1:z:0)batch_normalization_670/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_670/LeakyRelu	LeakyRelu+batch_normalization_670/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_740/MatMul/ReadVariableOpReadVariableOp(dense_740_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0
dense_740/MatMulMatMul'leaky_re_lu_670/LeakyRelu:activations:0'dense_740/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_740/BiasAdd/ReadVariableOpReadVariableOp)dense_740_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_740/BiasAddBiasAdddense_740/MatMul:product:0(dense_740/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
6batch_normalization_671/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_671/moments/meanMeandense_740/BiasAdd:output:0?batch_normalization_671/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
,batch_normalization_671/moments/StopGradientStopGradient-batch_normalization_671/moments/mean:output:0*
T0*
_output_shapes

:AË
1batch_normalization_671/moments/SquaredDifferenceSquaredDifferencedense_740/BiasAdd:output:05batch_normalization_671/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
:batch_normalization_671/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_671/moments/varianceMean5batch_normalization_671/moments/SquaredDifference:z:0Cbatch_normalization_671/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
'batch_normalization_671/moments/SqueezeSqueeze-batch_normalization_671/moments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 £
)batch_normalization_671/moments/Squeeze_1Squeeze1batch_normalization_671/moments/variance:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 r
-batch_normalization_671/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_671/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_671_assignmovingavg_readvariableop_resource*
_output_shapes
:A*
dtype0É
+batch_normalization_671/AssignMovingAvg/subSub>batch_normalization_671/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_671/moments/Squeeze:output:0*
T0*
_output_shapes
:AÀ
+batch_normalization_671/AssignMovingAvg/mulMul/batch_normalization_671/AssignMovingAvg/sub:z:06batch_normalization_671/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A
'batch_normalization_671/AssignMovingAvgAssignSubVariableOp?batch_normalization_671_assignmovingavg_readvariableop_resource/batch_normalization_671/AssignMovingAvg/mul:z:07^batch_normalization_671/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_671/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_671/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_671_assignmovingavg_1_readvariableop_resource*
_output_shapes
:A*
dtype0Ï
-batch_normalization_671/AssignMovingAvg_1/subSub@batch_normalization_671/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_671/moments/Squeeze_1:output:0*
T0*
_output_shapes
:AÆ
-batch_normalization_671/AssignMovingAvg_1/mulMul1batch_normalization_671/AssignMovingAvg_1/sub:z:08batch_normalization_671/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A
)batch_normalization_671/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_671_assignmovingavg_1_readvariableop_resource1batch_normalization_671/AssignMovingAvg_1/mul:z:09^batch_normalization_671/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_671/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_671/batchnorm/addAddV22batch_normalization_671/moments/Squeeze_1:output:00batch_normalization_671/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
'batch_normalization_671/batchnorm/RsqrtRsqrt)batch_normalization_671/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_671/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_671_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_671/batchnorm/mulMul+batch_normalization_671/batchnorm/Rsqrt:y:0<batch_normalization_671/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_671/batchnorm/mul_1Muldense_740/BiasAdd:output:0)batch_normalization_671/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA°
'batch_normalization_671/batchnorm/mul_2Mul0batch_normalization_671/moments/Squeeze:output:0)batch_normalization_671/batchnorm/mul:z:0*
T0*
_output_shapes
:A¦
0batch_normalization_671/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_671_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0¸
%batch_normalization_671/batchnorm/subSub8batch_normalization_671/batchnorm/ReadVariableOp:value:0+batch_normalization_671/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_671/batchnorm/add_1AddV2+batch_normalization_671/batchnorm/mul_1:z:0)batch_normalization_671/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_671/LeakyRelu	LeakyRelu+batch_normalization_671/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_741/MatMul/ReadVariableOpReadVariableOp(dense_741_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0
dense_741/MatMulMatMul'leaky_re_lu_671/LeakyRelu:activations:0'dense_741/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_741/BiasAdd/ReadVariableOpReadVariableOp)dense_741_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_741/BiasAddBiasAdddense_741/MatMul:product:0(dense_741/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
6batch_normalization_672/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_672/moments/meanMeandense_741/BiasAdd:output:0?batch_normalization_672/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
,batch_normalization_672/moments/StopGradientStopGradient-batch_normalization_672/moments/mean:output:0*
T0*
_output_shapes

:AË
1batch_normalization_672/moments/SquaredDifferenceSquaredDifferencedense_741/BiasAdd:output:05batch_normalization_672/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
:batch_normalization_672/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_672/moments/varianceMean5batch_normalization_672/moments/SquaredDifference:z:0Cbatch_normalization_672/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
'batch_normalization_672/moments/SqueezeSqueeze-batch_normalization_672/moments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 £
)batch_normalization_672/moments/Squeeze_1Squeeze1batch_normalization_672/moments/variance:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 r
-batch_normalization_672/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_672/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_672_assignmovingavg_readvariableop_resource*
_output_shapes
:A*
dtype0É
+batch_normalization_672/AssignMovingAvg/subSub>batch_normalization_672/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_672/moments/Squeeze:output:0*
T0*
_output_shapes
:AÀ
+batch_normalization_672/AssignMovingAvg/mulMul/batch_normalization_672/AssignMovingAvg/sub:z:06batch_normalization_672/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A
'batch_normalization_672/AssignMovingAvgAssignSubVariableOp?batch_normalization_672_assignmovingavg_readvariableop_resource/batch_normalization_672/AssignMovingAvg/mul:z:07^batch_normalization_672/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_672/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_672/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_672_assignmovingavg_1_readvariableop_resource*
_output_shapes
:A*
dtype0Ï
-batch_normalization_672/AssignMovingAvg_1/subSub@batch_normalization_672/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_672/moments/Squeeze_1:output:0*
T0*
_output_shapes
:AÆ
-batch_normalization_672/AssignMovingAvg_1/mulMul1batch_normalization_672/AssignMovingAvg_1/sub:z:08batch_normalization_672/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A
)batch_normalization_672/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_672_assignmovingavg_1_readvariableop_resource1batch_normalization_672/AssignMovingAvg_1/mul:z:09^batch_normalization_672/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_672/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_672/batchnorm/addAddV22batch_normalization_672/moments/Squeeze_1:output:00batch_normalization_672/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
'batch_normalization_672/batchnorm/RsqrtRsqrt)batch_normalization_672/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_672/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_672_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_672/batchnorm/mulMul+batch_normalization_672/batchnorm/Rsqrt:y:0<batch_normalization_672/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_672/batchnorm/mul_1Muldense_741/BiasAdd:output:0)batch_normalization_672/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA°
'batch_normalization_672/batchnorm/mul_2Mul0batch_normalization_672/moments/Squeeze:output:0)batch_normalization_672/batchnorm/mul:z:0*
T0*
_output_shapes
:A¦
0batch_normalization_672/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_672_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0¸
%batch_normalization_672/batchnorm/subSub8batch_normalization_672/batchnorm/ReadVariableOp:value:0+batch_normalization_672/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_672/batchnorm/add_1AddV2+batch_normalization_672/batchnorm/mul_1:z:0)batch_normalization_672/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_672/LeakyRelu	LeakyRelu+batch_normalization_672/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_742/MatMul/ReadVariableOpReadVariableOp(dense_742_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0
dense_742/MatMulMatMul'leaky_re_lu_672/LeakyRelu:activations:0'dense_742/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_742/BiasAdd/ReadVariableOpReadVariableOp)dense_742_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_742/BiasAddBiasAdddense_742/MatMul:product:0(dense_742/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
6batch_normalization_673/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_673/moments/meanMeandense_742/BiasAdd:output:0?batch_normalization_673/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
,batch_normalization_673/moments/StopGradientStopGradient-batch_normalization_673/moments/mean:output:0*
T0*
_output_shapes

:AË
1batch_normalization_673/moments/SquaredDifferenceSquaredDifferencedense_742/BiasAdd:output:05batch_normalization_673/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
:batch_normalization_673/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_673/moments/varianceMean5batch_normalization_673/moments/SquaredDifference:z:0Cbatch_normalization_673/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:A*
	keep_dims(
'batch_normalization_673/moments/SqueezeSqueeze-batch_normalization_673/moments/mean:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 £
)batch_normalization_673/moments/Squeeze_1Squeeze1batch_normalization_673/moments/variance:output:0*
T0*
_output_shapes
:A*
squeeze_dims
 r
-batch_normalization_673/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_673/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_673_assignmovingavg_readvariableop_resource*
_output_shapes
:A*
dtype0É
+batch_normalization_673/AssignMovingAvg/subSub>batch_normalization_673/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_673/moments/Squeeze:output:0*
T0*
_output_shapes
:AÀ
+batch_normalization_673/AssignMovingAvg/mulMul/batch_normalization_673/AssignMovingAvg/sub:z:06batch_normalization_673/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:A
'batch_normalization_673/AssignMovingAvgAssignSubVariableOp?batch_normalization_673_assignmovingavg_readvariableop_resource/batch_normalization_673/AssignMovingAvg/mul:z:07^batch_normalization_673/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_673/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_673/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_673_assignmovingavg_1_readvariableop_resource*
_output_shapes
:A*
dtype0Ï
-batch_normalization_673/AssignMovingAvg_1/subSub@batch_normalization_673/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_673/moments/Squeeze_1:output:0*
T0*
_output_shapes
:AÆ
-batch_normalization_673/AssignMovingAvg_1/mulMul1batch_normalization_673/AssignMovingAvg_1/sub:z:08batch_normalization_673/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:A
)batch_normalization_673/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_673_assignmovingavg_1_readvariableop_resource1batch_normalization_673/AssignMovingAvg_1/mul:z:09^batch_normalization_673/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_673/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_673/batchnorm/addAddV22batch_normalization_673/moments/Squeeze_1:output:00batch_normalization_673/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
'batch_normalization_673/batchnorm/RsqrtRsqrt)batch_normalization_673/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_673/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_673_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_673/batchnorm/mulMul+batch_normalization_673/batchnorm/Rsqrt:y:0<batch_normalization_673/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_673/batchnorm/mul_1Muldense_742/BiasAdd:output:0)batch_normalization_673/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA°
'batch_normalization_673/batchnorm/mul_2Mul0batch_normalization_673/moments/Squeeze:output:0)batch_normalization_673/batchnorm/mul:z:0*
T0*
_output_shapes
:A¦
0batch_normalization_673/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_673_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0¸
%batch_normalization_673/batchnorm/subSub8batch_normalization_673/batchnorm/ReadVariableOp:value:0+batch_normalization_673/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_673/batchnorm/add_1AddV2+batch_normalization_673/batchnorm/mul_1:z:0)batch_normalization_673/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_673/LeakyRelu	LeakyRelu+batch_normalization_673/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_743/MatMul/ReadVariableOpReadVariableOp(dense_743_matmul_readvariableop_resource*
_output_shapes

:A*
dtype0
dense_743/MatMulMatMul'leaky_re_lu_673/LeakyRelu:activations:0'dense_743/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_743/BiasAdd/ReadVariableOpReadVariableOp)dense_743_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_743/BiasAddBiasAdddense_743/MatMul:product:0(dense_743/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_743/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
NoOpNoOp(^batch_normalization_665/AssignMovingAvg7^batch_normalization_665/AssignMovingAvg/ReadVariableOp*^batch_normalization_665/AssignMovingAvg_19^batch_normalization_665/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_665/batchnorm/ReadVariableOp5^batch_normalization_665/batchnorm/mul/ReadVariableOp(^batch_normalization_666/AssignMovingAvg7^batch_normalization_666/AssignMovingAvg/ReadVariableOp*^batch_normalization_666/AssignMovingAvg_19^batch_normalization_666/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_666/batchnorm/ReadVariableOp5^batch_normalization_666/batchnorm/mul/ReadVariableOp(^batch_normalization_667/AssignMovingAvg7^batch_normalization_667/AssignMovingAvg/ReadVariableOp*^batch_normalization_667/AssignMovingAvg_19^batch_normalization_667/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_667/batchnorm/ReadVariableOp5^batch_normalization_667/batchnorm/mul/ReadVariableOp(^batch_normalization_668/AssignMovingAvg7^batch_normalization_668/AssignMovingAvg/ReadVariableOp*^batch_normalization_668/AssignMovingAvg_19^batch_normalization_668/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_668/batchnorm/ReadVariableOp5^batch_normalization_668/batchnorm/mul/ReadVariableOp(^batch_normalization_669/AssignMovingAvg7^batch_normalization_669/AssignMovingAvg/ReadVariableOp*^batch_normalization_669/AssignMovingAvg_19^batch_normalization_669/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_669/batchnorm/ReadVariableOp5^batch_normalization_669/batchnorm/mul/ReadVariableOp(^batch_normalization_670/AssignMovingAvg7^batch_normalization_670/AssignMovingAvg/ReadVariableOp*^batch_normalization_670/AssignMovingAvg_19^batch_normalization_670/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_670/batchnorm/ReadVariableOp5^batch_normalization_670/batchnorm/mul/ReadVariableOp(^batch_normalization_671/AssignMovingAvg7^batch_normalization_671/AssignMovingAvg/ReadVariableOp*^batch_normalization_671/AssignMovingAvg_19^batch_normalization_671/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_671/batchnorm/ReadVariableOp5^batch_normalization_671/batchnorm/mul/ReadVariableOp(^batch_normalization_672/AssignMovingAvg7^batch_normalization_672/AssignMovingAvg/ReadVariableOp*^batch_normalization_672/AssignMovingAvg_19^batch_normalization_672/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_672/batchnorm/ReadVariableOp5^batch_normalization_672/batchnorm/mul/ReadVariableOp(^batch_normalization_673/AssignMovingAvg7^batch_normalization_673/AssignMovingAvg/ReadVariableOp*^batch_normalization_673/AssignMovingAvg_19^batch_normalization_673/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_673/batchnorm/ReadVariableOp5^batch_normalization_673/batchnorm/mul/ReadVariableOp!^dense_734/BiasAdd/ReadVariableOp ^dense_734/MatMul/ReadVariableOp!^dense_735/BiasAdd/ReadVariableOp ^dense_735/MatMul/ReadVariableOp!^dense_736/BiasAdd/ReadVariableOp ^dense_736/MatMul/ReadVariableOp!^dense_737/BiasAdd/ReadVariableOp ^dense_737/MatMul/ReadVariableOp!^dense_738/BiasAdd/ReadVariableOp ^dense_738/MatMul/ReadVariableOp!^dense_739/BiasAdd/ReadVariableOp ^dense_739/MatMul/ReadVariableOp!^dense_740/BiasAdd/ReadVariableOp ^dense_740/MatMul/ReadVariableOp!^dense_741/BiasAdd/ReadVariableOp ^dense_741/MatMul/ReadVariableOp!^dense_742/BiasAdd/ReadVariableOp ^dense_742/MatMul/ReadVariableOp!^dense_743/BiasAdd/ReadVariableOp ^dense_743/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization_665/AssignMovingAvg'batch_normalization_665/AssignMovingAvg2p
6batch_normalization_665/AssignMovingAvg/ReadVariableOp6batch_normalization_665/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_665/AssignMovingAvg_1)batch_normalization_665/AssignMovingAvg_12t
8batch_normalization_665/AssignMovingAvg_1/ReadVariableOp8batch_normalization_665/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_665/batchnorm/ReadVariableOp0batch_normalization_665/batchnorm/ReadVariableOp2l
4batch_normalization_665/batchnorm/mul/ReadVariableOp4batch_normalization_665/batchnorm/mul/ReadVariableOp2R
'batch_normalization_666/AssignMovingAvg'batch_normalization_666/AssignMovingAvg2p
6batch_normalization_666/AssignMovingAvg/ReadVariableOp6batch_normalization_666/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_666/AssignMovingAvg_1)batch_normalization_666/AssignMovingAvg_12t
8batch_normalization_666/AssignMovingAvg_1/ReadVariableOp8batch_normalization_666/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_666/batchnorm/ReadVariableOp0batch_normalization_666/batchnorm/ReadVariableOp2l
4batch_normalization_666/batchnorm/mul/ReadVariableOp4batch_normalization_666/batchnorm/mul/ReadVariableOp2R
'batch_normalization_667/AssignMovingAvg'batch_normalization_667/AssignMovingAvg2p
6batch_normalization_667/AssignMovingAvg/ReadVariableOp6batch_normalization_667/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_667/AssignMovingAvg_1)batch_normalization_667/AssignMovingAvg_12t
8batch_normalization_667/AssignMovingAvg_1/ReadVariableOp8batch_normalization_667/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_667/batchnorm/ReadVariableOp0batch_normalization_667/batchnorm/ReadVariableOp2l
4batch_normalization_667/batchnorm/mul/ReadVariableOp4batch_normalization_667/batchnorm/mul/ReadVariableOp2R
'batch_normalization_668/AssignMovingAvg'batch_normalization_668/AssignMovingAvg2p
6batch_normalization_668/AssignMovingAvg/ReadVariableOp6batch_normalization_668/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_668/AssignMovingAvg_1)batch_normalization_668/AssignMovingAvg_12t
8batch_normalization_668/AssignMovingAvg_1/ReadVariableOp8batch_normalization_668/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_668/batchnorm/ReadVariableOp0batch_normalization_668/batchnorm/ReadVariableOp2l
4batch_normalization_668/batchnorm/mul/ReadVariableOp4batch_normalization_668/batchnorm/mul/ReadVariableOp2R
'batch_normalization_669/AssignMovingAvg'batch_normalization_669/AssignMovingAvg2p
6batch_normalization_669/AssignMovingAvg/ReadVariableOp6batch_normalization_669/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_669/AssignMovingAvg_1)batch_normalization_669/AssignMovingAvg_12t
8batch_normalization_669/AssignMovingAvg_1/ReadVariableOp8batch_normalization_669/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_669/batchnorm/ReadVariableOp0batch_normalization_669/batchnorm/ReadVariableOp2l
4batch_normalization_669/batchnorm/mul/ReadVariableOp4batch_normalization_669/batchnorm/mul/ReadVariableOp2R
'batch_normalization_670/AssignMovingAvg'batch_normalization_670/AssignMovingAvg2p
6batch_normalization_670/AssignMovingAvg/ReadVariableOp6batch_normalization_670/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_670/AssignMovingAvg_1)batch_normalization_670/AssignMovingAvg_12t
8batch_normalization_670/AssignMovingAvg_1/ReadVariableOp8batch_normalization_670/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_670/batchnorm/ReadVariableOp0batch_normalization_670/batchnorm/ReadVariableOp2l
4batch_normalization_670/batchnorm/mul/ReadVariableOp4batch_normalization_670/batchnorm/mul/ReadVariableOp2R
'batch_normalization_671/AssignMovingAvg'batch_normalization_671/AssignMovingAvg2p
6batch_normalization_671/AssignMovingAvg/ReadVariableOp6batch_normalization_671/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_671/AssignMovingAvg_1)batch_normalization_671/AssignMovingAvg_12t
8batch_normalization_671/AssignMovingAvg_1/ReadVariableOp8batch_normalization_671/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_671/batchnorm/ReadVariableOp0batch_normalization_671/batchnorm/ReadVariableOp2l
4batch_normalization_671/batchnorm/mul/ReadVariableOp4batch_normalization_671/batchnorm/mul/ReadVariableOp2R
'batch_normalization_672/AssignMovingAvg'batch_normalization_672/AssignMovingAvg2p
6batch_normalization_672/AssignMovingAvg/ReadVariableOp6batch_normalization_672/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_672/AssignMovingAvg_1)batch_normalization_672/AssignMovingAvg_12t
8batch_normalization_672/AssignMovingAvg_1/ReadVariableOp8batch_normalization_672/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_672/batchnorm/ReadVariableOp0batch_normalization_672/batchnorm/ReadVariableOp2l
4batch_normalization_672/batchnorm/mul/ReadVariableOp4batch_normalization_672/batchnorm/mul/ReadVariableOp2R
'batch_normalization_673/AssignMovingAvg'batch_normalization_673/AssignMovingAvg2p
6batch_normalization_673/AssignMovingAvg/ReadVariableOp6batch_normalization_673/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_673/AssignMovingAvg_1)batch_normalization_673/AssignMovingAvg_12t
8batch_normalization_673/AssignMovingAvg_1/ReadVariableOp8batch_normalization_673/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_673/batchnorm/ReadVariableOp0batch_normalization_673/batchnorm/ReadVariableOp2l
4batch_normalization_673/batchnorm/mul/ReadVariableOp4batch_normalization_673/batchnorm/mul/ReadVariableOp2D
 dense_734/BiasAdd/ReadVariableOp dense_734/BiasAdd/ReadVariableOp2B
dense_734/MatMul/ReadVariableOpdense_734/MatMul/ReadVariableOp2D
 dense_735/BiasAdd/ReadVariableOp dense_735/BiasAdd/ReadVariableOp2B
dense_735/MatMul/ReadVariableOpdense_735/MatMul/ReadVariableOp2D
 dense_736/BiasAdd/ReadVariableOp dense_736/BiasAdd/ReadVariableOp2B
dense_736/MatMul/ReadVariableOpdense_736/MatMul/ReadVariableOp2D
 dense_737/BiasAdd/ReadVariableOp dense_737/BiasAdd/ReadVariableOp2B
dense_737/MatMul/ReadVariableOpdense_737/MatMul/ReadVariableOp2D
 dense_738/BiasAdd/ReadVariableOp dense_738/BiasAdd/ReadVariableOp2B
dense_738/MatMul/ReadVariableOpdense_738/MatMul/ReadVariableOp2D
 dense_739/BiasAdd/ReadVariableOp dense_739/BiasAdd/ReadVariableOp2B
dense_739/MatMul/ReadVariableOpdense_739/MatMul/ReadVariableOp2D
 dense_740/BiasAdd/ReadVariableOp dense_740/BiasAdd/ReadVariableOp2B
dense_740/MatMul/ReadVariableOpdense_740/MatMul/ReadVariableOp2D
 dense_741/BiasAdd/ReadVariableOp dense_741/BiasAdd/ReadVariableOp2B
dense_741/MatMul/ReadVariableOpdense_741/MatMul/ReadVariableOp2D
 dense_742/BiasAdd/ReadVariableOp dense_742/BiasAdd/ReadVariableOp2B
dense_742/MatMul/ReadVariableOpdense_742/MatMul/ReadVariableOp2D
 dense_743/BiasAdd/ReadVariableOp dense_743/BiasAdd/ReadVariableOp2B
dense_743/MatMul/ReadVariableOpdense_743/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_668_layer_call_and_return_conditional_losses_736565

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
K__inference_leaky_re_lu_668_layer_call_and_return_conditional_losses_733871

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
E__inference_dense_738_layer_call_and_return_conditional_losses_736584

inputs0
matmul_readvariableop_resource:aA-
biasadd_readvariableop_resource:A
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:aA*
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
:ÿÿÿÿÿÿÿÿÿa: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_672_layer_call_fn_736937

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_672_layer_call_and_return_conditional_losses_733638o
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
Ä

*__inference_dense_735_layer_call_fn_736247

inputs
unknown:SS
	unknown_0:S
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_735_layer_call_and_return_conditional_losses_733787o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿS: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_666_layer_call_and_return_conditional_losses_736337

inputs5
'assignmovingavg_readvariableop_resource:S7
)assignmovingavg_1_readvariableop_resource:S3
%batchnorm_mul_readvariableop_resource:S/
!batchnorm_readvariableop_resource:S
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:S*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:S
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:S*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:S*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:S*
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
:S*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Sx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:S¬
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
:S*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:S~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:S´
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
:SP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:S~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:S*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Sc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Sv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:S*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Sr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿS: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_669_layer_call_and_return_conditional_losses_736630

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
«
L
0__inference_leaky_re_lu_673_layer_call_fn_737105

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
:ÿÿÿÿÿÿÿÿÿA* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_673_layer_call_and_return_conditional_losses_734031`
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
È	
ö
E__inference_dense_740_layer_call_and_return_conditional_losses_733947

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
¡
@
!__inference__wrapped_model_732993
normalization_69_input(
$sequential_69_normalization_69_sub_y)
%sequential_69_normalization_69_sqrt_xH
6sequential_69_dense_734_matmul_readvariableop_resource:SE
7sequential_69_dense_734_biasadd_readvariableop_resource:SU
Gsequential_69_batch_normalization_665_batchnorm_readvariableop_resource:SY
Ksequential_69_batch_normalization_665_batchnorm_mul_readvariableop_resource:SW
Isequential_69_batch_normalization_665_batchnorm_readvariableop_1_resource:SW
Isequential_69_batch_normalization_665_batchnorm_readvariableop_2_resource:SH
6sequential_69_dense_735_matmul_readvariableop_resource:SSE
7sequential_69_dense_735_biasadd_readvariableop_resource:SU
Gsequential_69_batch_normalization_666_batchnorm_readvariableop_resource:SY
Ksequential_69_batch_normalization_666_batchnorm_mul_readvariableop_resource:SW
Isequential_69_batch_normalization_666_batchnorm_readvariableop_1_resource:SW
Isequential_69_batch_normalization_666_batchnorm_readvariableop_2_resource:SH
6sequential_69_dense_736_matmul_readvariableop_resource:SaE
7sequential_69_dense_736_biasadd_readvariableop_resource:aU
Gsequential_69_batch_normalization_667_batchnorm_readvariableop_resource:aY
Ksequential_69_batch_normalization_667_batchnorm_mul_readvariableop_resource:aW
Isequential_69_batch_normalization_667_batchnorm_readvariableop_1_resource:aW
Isequential_69_batch_normalization_667_batchnorm_readvariableop_2_resource:aH
6sequential_69_dense_737_matmul_readvariableop_resource:aaE
7sequential_69_dense_737_biasadd_readvariableop_resource:aU
Gsequential_69_batch_normalization_668_batchnorm_readvariableop_resource:aY
Ksequential_69_batch_normalization_668_batchnorm_mul_readvariableop_resource:aW
Isequential_69_batch_normalization_668_batchnorm_readvariableop_1_resource:aW
Isequential_69_batch_normalization_668_batchnorm_readvariableop_2_resource:aH
6sequential_69_dense_738_matmul_readvariableop_resource:aAE
7sequential_69_dense_738_biasadd_readvariableop_resource:AU
Gsequential_69_batch_normalization_669_batchnorm_readvariableop_resource:AY
Ksequential_69_batch_normalization_669_batchnorm_mul_readvariableop_resource:AW
Isequential_69_batch_normalization_669_batchnorm_readvariableop_1_resource:AW
Isequential_69_batch_normalization_669_batchnorm_readvariableop_2_resource:AH
6sequential_69_dense_739_matmul_readvariableop_resource:AAE
7sequential_69_dense_739_biasadd_readvariableop_resource:AU
Gsequential_69_batch_normalization_670_batchnorm_readvariableop_resource:AY
Ksequential_69_batch_normalization_670_batchnorm_mul_readvariableop_resource:AW
Isequential_69_batch_normalization_670_batchnorm_readvariableop_1_resource:AW
Isequential_69_batch_normalization_670_batchnorm_readvariableop_2_resource:AH
6sequential_69_dense_740_matmul_readvariableop_resource:AAE
7sequential_69_dense_740_biasadd_readvariableop_resource:AU
Gsequential_69_batch_normalization_671_batchnorm_readvariableop_resource:AY
Ksequential_69_batch_normalization_671_batchnorm_mul_readvariableop_resource:AW
Isequential_69_batch_normalization_671_batchnorm_readvariableop_1_resource:AW
Isequential_69_batch_normalization_671_batchnorm_readvariableop_2_resource:AH
6sequential_69_dense_741_matmul_readvariableop_resource:AAE
7sequential_69_dense_741_biasadd_readvariableop_resource:AU
Gsequential_69_batch_normalization_672_batchnorm_readvariableop_resource:AY
Ksequential_69_batch_normalization_672_batchnorm_mul_readvariableop_resource:AW
Isequential_69_batch_normalization_672_batchnorm_readvariableop_1_resource:AW
Isequential_69_batch_normalization_672_batchnorm_readvariableop_2_resource:AH
6sequential_69_dense_742_matmul_readvariableop_resource:AAE
7sequential_69_dense_742_biasadd_readvariableop_resource:AU
Gsequential_69_batch_normalization_673_batchnorm_readvariableop_resource:AY
Ksequential_69_batch_normalization_673_batchnorm_mul_readvariableop_resource:AW
Isequential_69_batch_normalization_673_batchnorm_readvariableop_1_resource:AW
Isequential_69_batch_normalization_673_batchnorm_readvariableop_2_resource:AH
6sequential_69_dense_743_matmul_readvariableop_resource:AE
7sequential_69_dense_743_biasadd_readvariableop_resource:
identity¢>sequential_69/batch_normalization_665/batchnorm/ReadVariableOp¢@sequential_69/batch_normalization_665/batchnorm/ReadVariableOp_1¢@sequential_69/batch_normalization_665/batchnorm/ReadVariableOp_2¢Bsequential_69/batch_normalization_665/batchnorm/mul/ReadVariableOp¢>sequential_69/batch_normalization_666/batchnorm/ReadVariableOp¢@sequential_69/batch_normalization_666/batchnorm/ReadVariableOp_1¢@sequential_69/batch_normalization_666/batchnorm/ReadVariableOp_2¢Bsequential_69/batch_normalization_666/batchnorm/mul/ReadVariableOp¢>sequential_69/batch_normalization_667/batchnorm/ReadVariableOp¢@sequential_69/batch_normalization_667/batchnorm/ReadVariableOp_1¢@sequential_69/batch_normalization_667/batchnorm/ReadVariableOp_2¢Bsequential_69/batch_normalization_667/batchnorm/mul/ReadVariableOp¢>sequential_69/batch_normalization_668/batchnorm/ReadVariableOp¢@sequential_69/batch_normalization_668/batchnorm/ReadVariableOp_1¢@sequential_69/batch_normalization_668/batchnorm/ReadVariableOp_2¢Bsequential_69/batch_normalization_668/batchnorm/mul/ReadVariableOp¢>sequential_69/batch_normalization_669/batchnorm/ReadVariableOp¢@sequential_69/batch_normalization_669/batchnorm/ReadVariableOp_1¢@sequential_69/batch_normalization_669/batchnorm/ReadVariableOp_2¢Bsequential_69/batch_normalization_669/batchnorm/mul/ReadVariableOp¢>sequential_69/batch_normalization_670/batchnorm/ReadVariableOp¢@sequential_69/batch_normalization_670/batchnorm/ReadVariableOp_1¢@sequential_69/batch_normalization_670/batchnorm/ReadVariableOp_2¢Bsequential_69/batch_normalization_670/batchnorm/mul/ReadVariableOp¢>sequential_69/batch_normalization_671/batchnorm/ReadVariableOp¢@sequential_69/batch_normalization_671/batchnorm/ReadVariableOp_1¢@sequential_69/batch_normalization_671/batchnorm/ReadVariableOp_2¢Bsequential_69/batch_normalization_671/batchnorm/mul/ReadVariableOp¢>sequential_69/batch_normalization_672/batchnorm/ReadVariableOp¢@sequential_69/batch_normalization_672/batchnorm/ReadVariableOp_1¢@sequential_69/batch_normalization_672/batchnorm/ReadVariableOp_2¢Bsequential_69/batch_normalization_672/batchnorm/mul/ReadVariableOp¢>sequential_69/batch_normalization_673/batchnorm/ReadVariableOp¢@sequential_69/batch_normalization_673/batchnorm/ReadVariableOp_1¢@sequential_69/batch_normalization_673/batchnorm/ReadVariableOp_2¢Bsequential_69/batch_normalization_673/batchnorm/mul/ReadVariableOp¢.sequential_69/dense_734/BiasAdd/ReadVariableOp¢-sequential_69/dense_734/MatMul/ReadVariableOp¢.sequential_69/dense_735/BiasAdd/ReadVariableOp¢-sequential_69/dense_735/MatMul/ReadVariableOp¢.sequential_69/dense_736/BiasAdd/ReadVariableOp¢-sequential_69/dense_736/MatMul/ReadVariableOp¢.sequential_69/dense_737/BiasAdd/ReadVariableOp¢-sequential_69/dense_737/MatMul/ReadVariableOp¢.sequential_69/dense_738/BiasAdd/ReadVariableOp¢-sequential_69/dense_738/MatMul/ReadVariableOp¢.sequential_69/dense_739/BiasAdd/ReadVariableOp¢-sequential_69/dense_739/MatMul/ReadVariableOp¢.sequential_69/dense_740/BiasAdd/ReadVariableOp¢-sequential_69/dense_740/MatMul/ReadVariableOp¢.sequential_69/dense_741/BiasAdd/ReadVariableOp¢-sequential_69/dense_741/MatMul/ReadVariableOp¢.sequential_69/dense_742/BiasAdd/ReadVariableOp¢-sequential_69/dense_742/MatMul/ReadVariableOp¢.sequential_69/dense_743/BiasAdd/ReadVariableOp¢-sequential_69/dense_743/MatMul/ReadVariableOp
"sequential_69/normalization_69/subSubnormalization_69_input$sequential_69_normalization_69_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
#sequential_69/normalization_69/SqrtSqrt%sequential_69_normalization_69_sqrt_x*
T0*
_output_shapes

:m
(sequential_69/normalization_69/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¶
&sequential_69/normalization_69/MaximumMaximum'sequential_69/normalization_69/Sqrt:y:01sequential_69/normalization_69/Maximum/y:output:0*
T0*
_output_shapes

:·
&sequential_69/normalization_69/truedivRealDiv&sequential_69/normalization_69/sub:z:0*sequential_69/normalization_69/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_69/dense_734/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_734_matmul_readvariableop_resource*
_output_shapes

:S*
dtype0½
sequential_69/dense_734/MatMulMatMul*sequential_69/normalization_69/truediv:z:05sequential_69/dense_734/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS¢
.sequential_69/dense_734/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_734_biasadd_readvariableop_resource*
_output_shapes
:S*
dtype0¾
sequential_69/dense_734/BiasAddBiasAdd(sequential_69/dense_734/MatMul:product:06sequential_69/dense_734/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSÂ
>sequential_69/batch_normalization_665/batchnorm/ReadVariableOpReadVariableOpGsequential_69_batch_normalization_665_batchnorm_readvariableop_resource*
_output_shapes
:S*
dtype0z
5sequential_69/batch_normalization_665/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_69/batch_normalization_665/batchnorm/addAddV2Fsequential_69/batch_normalization_665/batchnorm/ReadVariableOp:value:0>sequential_69/batch_normalization_665/batchnorm/add/y:output:0*
T0*
_output_shapes
:S
5sequential_69/batch_normalization_665/batchnorm/RsqrtRsqrt7sequential_69/batch_normalization_665/batchnorm/add:z:0*
T0*
_output_shapes
:SÊ
Bsequential_69/batch_normalization_665/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_69_batch_normalization_665_batchnorm_mul_readvariableop_resource*
_output_shapes
:S*
dtype0æ
3sequential_69/batch_normalization_665/batchnorm/mulMul9sequential_69/batch_normalization_665/batchnorm/Rsqrt:y:0Jsequential_69/batch_normalization_665/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:SÑ
5sequential_69/batch_normalization_665/batchnorm/mul_1Mul(sequential_69/dense_734/BiasAdd:output:07sequential_69/batch_normalization_665/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSÆ
@sequential_69/batch_normalization_665/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_69_batch_normalization_665_batchnorm_readvariableop_1_resource*
_output_shapes
:S*
dtype0ä
5sequential_69/batch_normalization_665/batchnorm/mul_2MulHsequential_69/batch_normalization_665/batchnorm/ReadVariableOp_1:value:07sequential_69/batch_normalization_665/batchnorm/mul:z:0*
T0*
_output_shapes
:SÆ
@sequential_69/batch_normalization_665/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_69_batch_normalization_665_batchnorm_readvariableop_2_resource*
_output_shapes
:S*
dtype0ä
3sequential_69/batch_normalization_665/batchnorm/subSubHsequential_69/batch_normalization_665/batchnorm/ReadVariableOp_2:value:09sequential_69/batch_normalization_665/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Sä
5sequential_69/batch_normalization_665/batchnorm/add_1AddV29sequential_69/batch_normalization_665/batchnorm/mul_1:z:07sequential_69/batch_normalization_665/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS¨
'sequential_69/leaky_re_lu_665/LeakyRelu	LeakyRelu9sequential_69/batch_normalization_665/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
alpha%>¤
-sequential_69/dense_735/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_735_matmul_readvariableop_resource*
_output_shapes

:SS*
dtype0È
sequential_69/dense_735/MatMulMatMul5sequential_69/leaky_re_lu_665/LeakyRelu:activations:05sequential_69/dense_735/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS¢
.sequential_69/dense_735/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_735_biasadd_readvariableop_resource*
_output_shapes
:S*
dtype0¾
sequential_69/dense_735/BiasAddBiasAdd(sequential_69/dense_735/MatMul:product:06sequential_69/dense_735/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSÂ
>sequential_69/batch_normalization_666/batchnorm/ReadVariableOpReadVariableOpGsequential_69_batch_normalization_666_batchnorm_readvariableop_resource*
_output_shapes
:S*
dtype0z
5sequential_69/batch_normalization_666/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_69/batch_normalization_666/batchnorm/addAddV2Fsequential_69/batch_normalization_666/batchnorm/ReadVariableOp:value:0>sequential_69/batch_normalization_666/batchnorm/add/y:output:0*
T0*
_output_shapes
:S
5sequential_69/batch_normalization_666/batchnorm/RsqrtRsqrt7sequential_69/batch_normalization_666/batchnorm/add:z:0*
T0*
_output_shapes
:SÊ
Bsequential_69/batch_normalization_666/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_69_batch_normalization_666_batchnorm_mul_readvariableop_resource*
_output_shapes
:S*
dtype0æ
3sequential_69/batch_normalization_666/batchnorm/mulMul9sequential_69/batch_normalization_666/batchnorm/Rsqrt:y:0Jsequential_69/batch_normalization_666/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:SÑ
5sequential_69/batch_normalization_666/batchnorm/mul_1Mul(sequential_69/dense_735/BiasAdd:output:07sequential_69/batch_normalization_666/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSÆ
@sequential_69/batch_normalization_666/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_69_batch_normalization_666_batchnorm_readvariableop_1_resource*
_output_shapes
:S*
dtype0ä
5sequential_69/batch_normalization_666/batchnorm/mul_2MulHsequential_69/batch_normalization_666/batchnorm/ReadVariableOp_1:value:07sequential_69/batch_normalization_666/batchnorm/mul:z:0*
T0*
_output_shapes
:SÆ
@sequential_69/batch_normalization_666/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_69_batch_normalization_666_batchnorm_readvariableop_2_resource*
_output_shapes
:S*
dtype0ä
3sequential_69/batch_normalization_666/batchnorm/subSubHsequential_69/batch_normalization_666/batchnorm/ReadVariableOp_2:value:09sequential_69/batch_normalization_666/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Sä
5sequential_69/batch_normalization_666/batchnorm/add_1AddV29sequential_69/batch_normalization_666/batchnorm/mul_1:z:07sequential_69/batch_normalization_666/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS¨
'sequential_69/leaky_re_lu_666/LeakyRelu	LeakyRelu9sequential_69/batch_normalization_666/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
alpha%>¤
-sequential_69/dense_736/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_736_matmul_readvariableop_resource*
_output_shapes

:Sa*
dtype0È
sequential_69/dense_736/MatMulMatMul5sequential_69/leaky_re_lu_666/LeakyRelu:activations:05sequential_69/dense_736/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¢
.sequential_69/dense_736/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_736_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0¾
sequential_69/dense_736/BiasAddBiasAdd(sequential_69/dense_736/MatMul:product:06sequential_69/dense_736/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaÂ
>sequential_69/batch_normalization_667/batchnorm/ReadVariableOpReadVariableOpGsequential_69_batch_normalization_667_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0z
5sequential_69/batch_normalization_667/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_69/batch_normalization_667/batchnorm/addAddV2Fsequential_69/batch_normalization_667/batchnorm/ReadVariableOp:value:0>sequential_69/batch_normalization_667/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
5sequential_69/batch_normalization_667/batchnorm/RsqrtRsqrt7sequential_69/batch_normalization_667/batchnorm/add:z:0*
T0*
_output_shapes
:aÊ
Bsequential_69/batch_normalization_667/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_69_batch_normalization_667_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0æ
3sequential_69/batch_normalization_667/batchnorm/mulMul9sequential_69/batch_normalization_667/batchnorm/Rsqrt:y:0Jsequential_69/batch_normalization_667/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:aÑ
5sequential_69/batch_normalization_667/batchnorm/mul_1Mul(sequential_69/dense_736/BiasAdd:output:07sequential_69/batch_normalization_667/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaÆ
@sequential_69/batch_normalization_667/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_69_batch_normalization_667_batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0ä
5sequential_69/batch_normalization_667/batchnorm/mul_2MulHsequential_69/batch_normalization_667/batchnorm/ReadVariableOp_1:value:07sequential_69/batch_normalization_667/batchnorm/mul:z:0*
T0*
_output_shapes
:aÆ
@sequential_69/batch_normalization_667/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_69_batch_normalization_667_batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0ä
3sequential_69/batch_normalization_667/batchnorm/subSubHsequential_69/batch_normalization_667/batchnorm/ReadVariableOp_2:value:09sequential_69/batch_normalization_667/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aä
5sequential_69/batch_normalization_667/batchnorm/add_1AddV29sequential_69/batch_normalization_667/batchnorm/mul_1:z:07sequential_69/batch_normalization_667/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¨
'sequential_69/leaky_re_lu_667/LeakyRelu	LeakyRelu9sequential_69/batch_normalization_667/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>¤
-sequential_69/dense_737/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_737_matmul_readvariableop_resource*
_output_shapes

:aa*
dtype0È
sequential_69/dense_737/MatMulMatMul5sequential_69/leaky_re_lu_667/LeakyRelu:activations:05sequential_69/dense_737/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¢
.sequential_69/dense_737/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_737_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0¾
sequential_69/dense_737/BiasAddBiasAdd(sequential_69/dense_737/MatMul:product:06sequential_69/dense_737/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaÂ
>sequential_69/batch_normalization_668/batchnorm/ReadVariableOpReadVariableOpGsequential_69_batch_normalization_668_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0z
5sequential_69/batch_normalization_668/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_69/batch_normalization_668/batchnorm/addAddV2Fsequential_69/batch_normalization_668/batchnorm/ReadVariableOp:value:0>sequential_69/batch_normalization_668/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
5sequential_69/batch_normalization_668/batchnorm/RsqrtRsqrt7sequential_69/batch_normalization_668/batchnorm/add:z:0*
T0*
_output_shapes
:aÊ
Bsequential_69/batch_normalization_668/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_69_batch_normalization_668_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0æ
3sequential_69/batch_normalization_668/batchnorm/mulMul9sequential_69/batch_normalization_668/batchnorm/Rsqrt:y:0Jsequential_69/batch_normalization_668/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:aÑ
5sequential_69/batch_normalization_668/batchnorm/mul_1Mul(sequential_69/dense_737/BiasAdd:output:07sequential_69/batch_normalization_668/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaÆ
@sequential_69/batch_normalization_668/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_69_batch_normalization_668_batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0ä
5sequential_69/batch_normalization_668/batchnorm/mul_2MulHsequential_69/batch_normalization_668/batchnorm/ReadVariableOp_1:value:07sequential_69/batch_normalization_668/batchnorm/mul:z:0*
T0*
_output_shapes
:aÆ
@sequential_69/batch_normalization_668/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_69_batch_normalization_668_batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0ä
3sequential_69/batch_normalization_668/batchnorm/subSubHsequential_69/batch_normalization_668/batchnorm/ReadVariableOp_2:value:09sequential_69/batch_normalization_668/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aä
5sequential_69/batch_normalization_668/batchnorm/add_1AddV29sequential_69/batch_normalization_668/batchnorm/mul_1:z:07sequential_69/batch_normalization_668/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¨
'sequential_69/leaky_re_lu_668/LeakyRelu	LeakyRelu9sequential_69/batch_normalization_668/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>¤
-sequential_69/dense_738/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_738_matmul_readvariableop_resource*
_output_shapes

:aA*
dtype0È
sequential_69/dense_738/MatMulMatMul5sequential_69/leaky_re_lu_668/LeakyRelu:activations:05sequential_69/dense_738/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¢
.sequential_69/dense_738/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_738_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0¾
sequential_69/dense_738/BiasAddBiasAdd(sequential_69/dense_738/MatMul:product:06sequential_69/dense_738/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÂ
>sequential_69/batch_normalization_669/batchnorm/ReadVariableOpReadVariableOpGsequential_69_batch_normalization_669_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0z
5sequential_69/batch_normalization_669/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_69/batch_normalization_669/batchnorm/addAddV2Fsequential_69/batch_normalization_669/batchnorm/ReadVariableOp:value:0>sequential_69/batch_normalization_669/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
5sequential_69/batch_normalization_669/batchnorm/RsqrtRsqrt7sequential_69/batch_normalization_669/batchnorm/add:z:0*
T0*
_output_shapes
:AÊ
Bsequential_69/batch_normalization_669/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_69_batch_normalization_669_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0æ
3sequential_69/batch_normalization_669/batchnorm/mulMul9sequential_69/batch_normalization_669/batchnorm/Rsqrt:y:0Jsequential_69/batch_normalization_669/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:AÑ
5sequential_69/batch_normalization_669/batchnorm/mul_1Mul(sequential_69/dense_738/BiasAdd:output:07sequential_69/batch_normalization_669/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÆ
@sequential_69/batch_normalization_669/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_69_batch_normalization_669_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0ä
5sequential_69/batch_normalization_669/batchnorm/mul_2MulHsequential_69/batch_normalization_669/batchnorm/ReadVariableOp_1:value:07sequential_69/batch_normalization_669/batchnorm/mul:z:0*
T0*
_output_shapes
:AÆ
@sequential_69/batch_normalization_669/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_69_batch_normalization_669_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0ä
3sequential_69/batch_normalization_669/batchnorm/subSubHsequential_69/batch_normalization_669/batchnorm/ReadVariableOp_2:value:09sequential_69/batch_normalization_669/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aä
5sequential_69/batch_normalization_669/batchnorm/add_1AddV29sequential_69/batch_normalization_669/batchnorm/mul_1:z:07sequential_69/batch_normalization_669/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¨
'sequential_69/leaky_re_lu_669/LeakyRelu	LeakyRelu9sequential_69/batch_normalization_669/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>¤
-sequential_69/dense_739/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_739_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0È
sequential_69/dense_739/MatMulMatMul5sequential_69/leaky_re_lu_669/LeakyRelu:activations:05sequential_69/dense_739/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¢
.sequential_69/dense_739/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_739_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0¾
sequential_69/dense_739/BiasAddBiasAdd(sequential_69/dense_739/MatMul:product:06sequential_69/dense_739/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÂ
>sequential_69/batch_normalization_670/batchnorm/ReadVariableOpReadVariableOpGsequential_69_batch_normalization_670_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0z
5sequential_69/batch_normalization_670/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_69/batch_normalization_670/batchnorm/addAddV2Fsequential_69/batch_normalization_670/batchnorm/ReadVariableOp:value:0>sequential_69/batch_normalization_670/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
5sequential_69/batch_normalization_670/batchnorm/RsqrtRsqrt7sequential_69/batch_normalization_670/batchnorm/add:z:0*
T0*
_output_shapes
:AÊ
Bsequential_69/batch_normalization_670/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_69_batch_normalization_670_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0æ
3sequential_69/batch_normalization_670/batchnorm/mulMul9sequential_69/batch_normalization_670/batchnorm/Rsqrt:y:0Jsequential_69/batch_normalization_670/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:AÑ
5sequential_69/batch_normalization_670/batchnorm/mul_1Mul(sequential_69/dense_739/BiasAdd:output:07sequential_69/batch_normalization_670/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÆ
@sequential_69/batch_normalization_670/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_69_batch_normalization_670_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0ä
5sequential_69/batch_normalization_670/batchnorm/mul_2MulHsequential_69/batch_normalization_670/batchnorm/ReadVariableOp_1:value:07sequential_69/batch_normalization_670/batchnorm/mul:z:0*
T0*
_output_shapes
:AÆ
@sequential_69/batch_normalization_670/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_69_batch_normalization_670_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0ä
3sequential_69/batch_normalization_670/batchnorm/subSubHsequential_69/batch_normalization_670/batchnorm/ReadVariableOp_2:value:09sequential_69/batch_normalization_670/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aä
5sequential_69/batch_normalization_670/batchnorm/add_1AddV29sequential_69/batch_normalization_670/batchnorm/mul_1:z:07sequential_69/batch_normalization_670/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¨
'sequential_69/leaky_re_lu_670/LeakyRelu	LeakyRelu9sequential_69/batch_normalization_670/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>¤
-sequential_69/dense_740/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_740_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0È
sequential_69/dense_740/MatMulMatMul5sequential_69/leaky_re_lu_670/LeakyRelu:activations:05sequential_69/dense_740/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¢
.sequential_69/dense_740/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_740_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0¾
sequential_69/dense_740/BiasAddBiasAdd(sequential_69/dense_740/MatMul:product:06sequential_69/dense_740/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÂ
>sequential_69/batch_normalization_671/batchnorm/ReadVariableOpReadVariableOpGsequential_69_batch_normalization_671_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0z
5sequential_69/batch_normalization_671/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_69/batch_normalization_671/batchnorm/addAddV2Fsequential_69/batch_normalization_671/batchnorm/ReadVariableOp:value:0>sequential_69/batch_normalization_671/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
5sequential_69/batch_normalization_671/batchnorm/RsqrtRsqrt7sequential_69/batch_normalization_671/batchnorm/add:z:0*
T0*
_output_shapes
:AÊ
Bsequential_69/batch_normalization_671/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_69_batch_normalization_671_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0æ
3sequential_69/batch_normalization_671/batchnorm/mulMul9sequential_69/batch_normalization_671/batchnorm/Rsqrt:y:0Jsequential_69/batch_normalization_671/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:AÑ
5sequential_69/batch_normalization_671/batchnorm/mul_1Mul(sequential_69/dense_740/BiasAdd:output:07sequential_69/batch_normalization_671/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÆ
@sequential_69/batch_normalization_671/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_69_batch_normalization_671_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0ä
5sequential_69/batch_normalization_671/batchnorm/mul_2MulHsequential_69/batch_normalization_671/batchnorm/ReadVariableOp_1:value:07sequential_69/batch_normalization_671/batchnorm/mul:z:0*
T0*
_output_shapes
:AÆ
@sequential_69/batch_normalization_671/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_69_batch_normalization_671_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0ä
3sequential_69/batch_normalization_671/batchnorm/subSubHsequential_69/batch_normalization_671/batchnorm/ReadVariableOp_2:value:09sequential_69/batch_normalization_671/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aä
5sequential_69/batch_normalization_671/batchnorm/add_1AddV29sequential_69/batch_normalization_671/batchnorm/mul_1:z:07sequential_69/batch_normalization_671/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¨
'sequential_69/leaky_re_lu_671/LeakyRelu	LeakyRelu9sequential_69/batch_normalization_671/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>¤
-sequential_69/dense_741/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_741_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0È
sequential_69/dense_741/MatMulMatMul5sequential_69/leaky_re_lu_671/LeakyRelu:activations:05sequential_69/dense_741/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¢
.sequential_69/dense_741/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_741_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0¾
sequential_69/dense_741/BiasAddBiasAdd(sequential_69/dense_741/MatMul:product:06sequential_69/dense_741/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÂ
>sequential_69/batch_normalization_672/batchnorm/ReadVariableOpReadVariableOpGsequential_69_batch_normalization_672_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0z
5sequential_69/batch_normalization_672/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_69/batch_normalization_672/batchnorm/addAddV2Fsequential_69/batch_normalization_672/batchnorm/ReadVariableOp:value:0>sequential_69/batch_normalization_672/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
5sequential_69/batch_normalization_672/batchnorm/RsqrtRsqrt7sequential_69/batch_normalization_672/batchnorm/add:z:0*
T0*
_output_shapes
:AÊ
Bsequential_69/batch_normalization_672/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_69_batch_normalization_672_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0æ
3sequential_69/batch_normalization_672/batchnorm/mulMul9sequential_69/batch_normalization_672/batchnorm/Rsqrt:y:0Jsequential_69/batch_normalization_672/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:AÑ
5sequential_69/batch_normalization_672/batchnorm/mul_1Mul(sequential_69/dense_741/BiasAdd:output:07sequential_69/batch_normalization_672/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÆ
@sequential_69/batch_normalization_672/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_69_batch_normalization_672_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0ä
5sequential_69/batch_normalization_672/batchnorm/mul_2MulHsequential_69/batch_normalization_672/batchnorm/ReadVariableOp_1:value:07sequential_69/batch_normalization_672/batchnorm/mul:z:0*
T0*
_output_shapes
:AÆ
@sequential_69/batch_normalization_672/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_69_batch_normalization_672_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0ä
3sequential_69/batch_normalization_672/batchnorm/subSubHsequential_69/batch_normalization_672/batchnorm/ReadVariableOp_2:value:09sequential_69/batch_normalization_672/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aä
5sequential_69/batch_normalization_672/batchnorm/add_1AddV29sequential_69/batch_normalization_672/batchnorm/mul_1:z:07sequential_69/batch_normalization_672/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¨
'sequential_69/leaky_re_lu_672/LeakyRelu	LeakyRelu9sequential_69/batch_normalization_672/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>¤
-sequential_69/dense_742/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_742_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0È
sequential_69/dense_742/MatMulMatMul5sequential_69/leaky_re_lu_672/LeakyRelu:activations:05sequential_69/dense_742/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¢
.sequential_69/dense_742/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_742_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0¾
sequential_69/dense_742/BiasAddBiasAdd(sequential_69/dense_742/MatMul:product:06sequential_69/dense_742/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÂ
>sequential_69/batch_normalization_673/batchnorm/ReadVariableOpReadVariableOpGsequential_69_batch_normalization_673_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0z
5sequential_69/batch_normalization_673/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_69/batch_normalization_673/batchnorm/addAddV2Fsequential_69/batch_normalization_673/batchnorm/ReadVariableOp:value:0>sequential_69/batch_normalization_673/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
5sequential_69/batch_normalization_673/batchnorm/RsqrtRsqrt7sequential_69/batch_normalization_673/batchnorm/add:z:0*
T0*
_output_shapes
:AÊ
Bsequential_69/batch_normalization_673/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_69_batch_normalization_673_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0æ
3sequential_69/batch_normalization_673/batchnorm/mulMul9sequential_69/batch_normalization_673/batchnorm/Rsqrt:y:0Jsequential_69/batch_normalization_673/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:AÑ
5sequential_69/batch_normalization_673/batchnorm/mul_1Mul(sequential_69/dense_742/BiasAdd:output:07sequential_69/batch_normalization_673/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAÆ
@sequential_69/batch_normalization_673/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_69_batch_normalization_673_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0ä
5sequential_69/batch_normalization_673/batchnorm/mul_2MulHsequential_69/batch_normalization_673/batchnorm/ReadVariableOp_1:value:07sequential_69/batch_normalization_673/batchnorm/mul:z:0*
T0*
_output_shapes
:AÆ
@sequential_69/batch_normalization_673/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_69_batch_normalization_673_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0ä
3sequential_69/batch_normalization_673/batchnorm/subSubHsequential_69/batch_normalization_673/batchnorm/ReadVariableOp_2:value:09sequential_69/batch_normalization_673/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aä
5sequential_69/batch_normalization_673/batchnorm/add_1AddV29sequential_69/batch_normalization_673/batchnorm/mul_1:z:07sequential_69/batch_normalization_673/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¨
'sequential_69/leaky_re_lu_673/LeakyRelu	LeakyRelu9sequential_69/batch_normalization_673/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>¤
-sequential_69/dense_743/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_743_matmul_readvariableop_resource*
_output_shapes

:A*
dtype0È
sequential_69/dense_743/MatMulMatMul5sequential_69/leaky_re_lu_673/LeakyRelu:activations:05sequential_69/dense_743/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_69/dense_743/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_743_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_69/dense_743/BiasAddBiasAdd(sequential_69/dense_743/MatMul:product:06sequential_69/dense_743/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_69/dense_743/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿü
NoOpNoOp?^sequential_69/batch_normalization_665/batchnorm/ReadVariableOpA^sequential_69/batch_normalization_665/batchnorm/ReadVariableOp_1A^sequential_69/batch_normalization_665/batchnorm/ReadVariableOp_2C^sequential_69/batch_normalization_665/batchnorm/mul/ReadVariableOp?^sequential_69/batch_normalization_666/batchnorm/ReadVariableOpA^sequential_69/batch_normalization_666/batchnorm/ReadVariableOp_1A^sequential_69/batch_normalization_666/batchnorm/ReadVariableOp_2C^sequential_69/batch_normalization_666/batchnorm/mul/ReadVariableOp?^sequential_69/batch_normalization_667/batchnorm/ReadVariableOpA^sequential_69/batch_normalization_667/batchnorm/ReadVariableOp_1A^sequential_69/batch_normalization_667/batchnorm/ReadVariableOp_2C^sequential_69/batch_normalization_667/batchnorm/mul/ReadVariableOp?^sequential_69/batch_normalization_668/batchnorm/ReadVariableOpA^sequential_69/batch_normalization_668/batchnorm/ReadVariableOp_1A^sequential_69/batch_normalization_668/batchnorm/ReadVariableOp_2C^sequential_69/batch_normalization_668/batchnorm/mul/ReadVariableOp?^sequential_69/batch_normalization_669/batchnorm/ReadVariableOpA^sequential_69/batch_normalization_669/batchnorm/ReadVariableOp_1A^sequential_69/batch_normalization_669/batchnorm/ReadVariableOp_2C^sequential_69/batch_normalization_669/batchnorm/mul/ReadVariableOp?^sequential_69/batch_normalization_670/batchnorm/ReadVariableOpA^sequential_69/batch_normalization_670/batchnorm/ReadVariableOp_1A^sequential_69/batch_normalization_670/batchnorm/ReadVariableOp_2C^sequential_69/batch_normalization_670/batchnorm/mul/ReadVariableOp?^sequential_69/batch_normalization_671/batchnorm/ReadVariableOpA^sequential_69/batch_normalization_671/batchnorm/ReadVariableOp_1A^sequential_69/batch_normalization_671/batchnorm/ReadVariableOp_2C^sequential_69/batch_normalization_671/batchnorm/mul/ReadVariableOp?^sequential_69/batch_normalization_672/batchnorm/ReadVariableOpA^sequential_69/batch_normalization_672/batchnorm/ReadVariableOp_1A^sequential_69/batch_normalization_672/batchnorm/ReadVariableOp_2C^sequential_69/batch_normalization_672/batchnorm/mul/ReadVariableOp?^sequential_69/batch_normalization_673/batchnorm/ReadVariableOpA^sequential_69/batch_normalization_673/batchnorm/ReadVariableOp_1A^sequential_69/batch_normalization_673/batchnorm/ReadVariableOp_2C^sequential_69/batch_normalization_673/batchnorm/mul/ReadVariableOp/^sequential_69/dense_734/BiasAdd/ReadVariableOp.^sequential_69/dense_734/MatMul/ReadVariableOp/^sequential_69/dense_735/BiasAdd/ReadVariableOp.^sequential_69/dense_735/MatMul/ReadVariableOp/^sequential_69/dense_736/BiasAdd/ReadVariableOp.^sequential_69/dense_736/MatMul/ReadVariableOp/^sequential_69/dense_737/BiasAdd/ReadVariableOp.^sequential_69/dense_737/MatMul/ReadVariableOp/^sequential_69/dense_738/BiasAdd/ReadVariableOp.^sequential_69/dense_738/MatMul/ReadVariableOp/^sequential_69/dense_739/BiasAdd/ReadVariableOp.^sequential_69/dense_739/MatMul/ReadVariableOp/^sequential_69/dense_740/BiasAdd/ReadVariableOp.^sequential_69/dense_740/MatMul/ReadVariableOp/^sequential_69/dense_741/BiasAdd/ReadVariableOp.^sequential_69/dense_741/MatMul/ReadVariableOp/^sequential_69/dense_742/BiasAdd/ReadVariableOp.^sequential_69/dense_742/MatMul/ReadVariableOp/^sequential_69/dense_743/BiasAdd/ReadVariableOp.^sequential_69/dense_743/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
>sequential_69/batch_normalization_665/batchnorm/ReadVariableOp>sequential_69/batch_normalization_665/batchnorm/ReadVariableOp2
@sequential_69/batch_normalization_665/batchnorm/ReadVariableOp_1@sequential_69/batch_normalization_665/batchnorm/ReadVariableOp_12
@sequential_69/batch_normalization_665/batchnorm/ReadVariableOp_2@sequential_69/batch_normalization_665/batchnorm/ReadVariableOp_22
Bsequential_69/batch_normalization_665/batchnorm/mul/ReadVariableOpBsequential_69/batch_normalization_665/batchnorm/mul/ReadVariableOp2
>sequential_69/batch_normalization_666/batchnorm/ReadVariableOp>sequential_69/batch_normalization_666/batchnorm/ReadVariableOp2
@sequential_69/batch_normalization_666/batchnorm/ReadVariableOp_1@sequential_69/batch_normalization_666/batchnorm/ReadVariableOp_12
@sequential_69/batch_normalization_666/batchnorm/ReadVariableOp_2@sequential_69/batch_normalization_666/batchnorm/ReadVariableOp_22
Bsequential_69/batch_normalization_666/batchnorm/mul/ReadVariableOpBsequential_69/batch_normalization_666/batchnorm/mul/ReadVariableOp2
>sequential_69/batch_normalization_667/batchnorm/ReadVariableOp>sequential_69/batch_normalization_667/batchnorm/ReadVariableOp2
@sequential_69/batch_normalization_667/batchnorm/ReadVariableOp_1@sequential_69/batch_normalization_667/batchnorm/ReadVariableOp_12
@sequential_69/batch_normalization_667/batchnorm/ReadVariableOp_2@sequential_69/batch_normalization_667/batchnorm/ReadVariableOp_22
Bsequential_69/batch_normalization_667/batchnorm/mul/ReadVariableOpBsequential_69/batch_normalization_667/batchnorm/mul/ReadVariableOp2
>sequential_69/batch_normalization_668/batchnorm/ReadVariableOp>sequential_69/batch_normalization_668/batchnorm/ReadVariableOp2
@sequential_69/batch_normalization_668/batchnorm/ReadVariableOp_1@sequential_69/batch_normalization_668/batchnorm/ReadVariableOp_12
@sequential_69/batch_normalization_668/batchnorm/ReadVariableOp_2@sequential_69/batch_normalization_668/batchnorm/ReadVariableOp_22
Bsequential_69/batch_normalization_668/batchnorm/mul/ReadVariableOpBsequential_69/batch_normalization_668/batchnorm/mul/ReadVariableOp2
>sequential_69/batch_normalization_669/batchnorm/ReadVariableOp>sequential_69/batch_normalization_669/batchnorm/ReadVariableOp2
@sequential_69/batch_normalization_669/batchnorm/ReadVariableOp_1@sequential_69/batch_normalization_669/batchnorm/ReadVariableOp_12
@sequential_69/batch_normalization_669/batchnorm/ReadVariableOp_2@sequential_69/batch_normalization_669/batchnorm/ReadVariableOp_22
Bsequential_69/batch_normalization_669/batchnorm/mul/ReadVariableOpBsequential_69/batch_normalization_669/batchnorm/mul/ReadVariableOp2
>sequential_69/batch_normalization_670/batchnorm/ReadVariableOp>sequential_69/batch_normalization_670/batchnorm/ReadVariableOp2
@sequential_69/batch_normalization_670/batchnorm/ReadVariableOp_1@sequential_69/batch_normalization_670/batchnorm/ReadVariableOp_12
@sequential_69/batch_normalization_670/batchnorm/ReadVariableOp_2@sequential_69/batch_normalization_670/batchnorm/ReadVariableOp_22
Bsequential_69/batch_normalization_670/batchnorm/mul/ReadVariableOpBsequential_69/batch_normalization_670/batchnorm/mul/ReadVariableOp2
>sequential_69/batch_normalization_671/batchnorm/ReadVariableOp>sequential_69/batch_normalization_671/batchnorm/ReadVariableOp2
@sequential_69/batch_normalization_671/batchnorm/ReadVariableOp_1@sequential_69/batch_normalization_671/batchnorm/ReadVariableOp_12
@sequential_69/batch_normalization_671/batchnorm/ReadVariableOp_2@sequential_69/batch_normalization_671/batchnorm/ReadVariableOp_22
Bsequential_69/batch_normalization_671/batchnorm/mul/ReadVariableOpBsequential_69/batch_normalization_671/batchnorm/mul/ReadVariableOp2
>sequential_69/batch_normalization_672/batchnorm/ReadVariableOp>sequential_69/batch_normalization_672/batchnorm/ReadVariableOp2
@sequential_69/batch_normalization_672/batchnorm/ReadVariableOp_1@sequential_69/batch_normalization_672/batchnorm/ReadVariableOp_12
@sequential_69/batch_normalization_672/batchnorm/ReadVariableOp_2@sequential_69/batch_normalization_672/batchnorm/ReadVariableOp_22
Bsequential_69/batch_normalization_672/batchnorm/mul/ReadVariableOpBsequential_69/batch_normalization_672/batchnorm/mul/ReadVariableOp2
>sequential_69/batch_normalization_673/batchnorm/ReadVariableOp>sequential_69/batch_normalization_673/batchnorm/ReadVariableOp2
@sequential_69/batch_normalization_673/batchnorm/ReadVariableOp_1@sequential_69/batch_normalization_673/batchnorm/ReadVariableOp_12
@sequential_69/batch_normalization_673/batchnorm/ReadVariableOp_2@sequential_69/batch_normalization_673/batchnorm/ReadVariableOp_22
Bsequential_69/batch_normalization_673/batchnorm/mul/ReadVariableOpBsequential_69/batch_normalization_673/batchnorm/mul/ReadVariableOp2`
.sequential_69/dense_734/BiasAdd/ReadVariableOp.sequential_69/dense_734/BiasAdd/ReadVariableOp2^
-sequential_69/dense_734/MatMul/ReadVariableOp-sequential_69/dense_734/MatMul/ReadVariableOp2`
.sequential_69/dense_735/BiasAdd/ReadVariableOp.sequential_69/dense_735/BiasAdd/ReadVariableOp2^
-sequential_69/dense_735/MatMul/ReadVariableOp-sequential_69/dense_735/MatMul/ReadVariableOp2`
.sequential_69/dense_736/BiasAdd/ReadVariableOp.sequential_69/dense_736/BiasAdd/ReadVariableOp2^
-sequential_69/dense_736/MatMul/ReadVariableOp-sequential_69/dense_736/MatMul/ReadVariableOp2`
.sequential_69/dense_737/BiasAdd/ReadVariableOp.sequential_69/dense_737/BiasAdd/ReadVariableOp2^
-sequential_69/dense_737/MatMul/ReadVariableOp-sequential_69/dense_737/MatMul/ReadVariableOp2`
.sequential_69/dense_738/BiasAdd/ReadVariableOp.sequential_69/dense_738/BiasAdd/ReadVariableOp2^
-sequential_69/dense_738/MatMul/ReadVariableOp-sequential_69/dense_738/MatMul/ReadVariableOp2`
.sequential_69/dense_739/BiasAdd/ReadVariableOp.sequential_69/dense_739/BiasAdd/ReadVariableOp2^
-sequential_69/dense_739/MatMul/ReadVariableOp-sequential_69/dense_739/MatMul/ReadVariableOp2`
.sequential_69/dense_740/BiasAdd/ReadVariableOp.sequential_69/dense_740/BiasAdd/ReadVariableOp2^
-sequential_69/dense_740/MatMul/ReadVariableOp-sequential_69/dense_740/MatMul/ReadVariableOp2`
.sequential_69/dense_741/BiasAdd/ReadVariableOp.sequential_69/dense_741/BiasAdd/ReadVariableOp2^
-sequential_69/dense_741/MatMul/ReadVariableOp-sequential_69/dense_741/MatMul/ReadVariableOp2`
.sequential_69/dense_742/BiasAdd/ReadVariableOp.sequential_69/dense_742/BiasAdd/ReadVariableOp2^
-sequential_69/dense_742/MatMul/ReadVariableOp-sequential_69/dense_742/MatMul/ReadVariableOp2`
.sequential_69/dense_743/BiasAdd/ReadVariableOp.sequential_69/dense_743/BiasAdd/ReadVariableOp2^
-sequential_69/dense_743/MatMul/ReadVariableOp-sequential_69/dense_743/MatMul/ReadVariableOp:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_69_input:$ 

_output_shapes

::$ 

_output_shapes

:
È	
ö
E__inference_dense_737_layer_call_and_return_conditional_losses_733851

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
E__inference_dense_742_layer_call_and_return_conditional_losses_737020

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
Ð
²
S__inference_batch_normalization_667_layer_call_and_return_conditional_losses_733181

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
S__inference_batch_normalization_673_layer_call_and_return_conditional_losses_737066

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
å
g
K__inference_leaky_re_lu_670_layer_call_and_return_conditional_losses_733935

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
%
ì
S__inference_batch_normalization_670_layer_call_and_return_conditional_losses_733474

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

â
.__inference_sequential_69_layer_call_fn_734169
normalization_69_input
unknown
	unknown_0
	unknown_1:S
	unknown_2:S
	unknown_3:S
	unknown_4:S
	unknown_5:S
	unknown_6:S
	unknown_7:SS
	unknown_8:S
	unknown_9:S

unknown_10:S

unknown_11:S

unknown_12:S

unknown_13:Sa

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

unknown_25:aA

unknown_26:A

unknown_27:A

unknown_28:A

unknown_29:A

unknown_30:A

unknown_31:AA

unknown_32:A

unknown_33:A

unknown_34:A

unknown_35:A

unknown_36:A

unknown_37:AA

unknown_38:A

unknown_39:A

unknown_40:A

unknown_41:A

unknown_42:A

unknown_43:AA

unknown_44:A

unknown_45:A

unknown_46:A

unknown_47:A

unknown_48:A

unknown_49:AA

unknown_50:A

unknown_51:A

unknown_52:A

unknown_53:A

unknown_54:A

unknown_55:A

unknown_56:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallnormalization_69_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_69_layer_call_and_return_conditional_losses_734050o
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
_user_specified_namenormalization_69_input:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_671_layer_call_and_return_conditional_losses_733967

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
Ä

*__inference_dense_738_layer_call_fn_736574

inputs
unknown:aA
	unknown_0:A
identity¢StatefulPartitionedCallÚ
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
GPU 2J 8 *N
fIRG
E__inference_dense_738_layer_call_and_return_conditional_losses_733883o
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
:ÿÿÿÿÿÿÿÿÿa: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_666_layer_call_and_return_conditional_losses_733099

inputs/
!batchnorm_readvariableop_resource:S3
%batchnorm_mul_readvariableop_resource:S1
#batchnorm_readvariableop_1_resource:S1
#batchnorm_readvariableop_2_resource:S
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:S*
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
:SP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:S~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:S*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Sc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:S*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Sz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:S*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Sr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿS: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
È	
ö
E__inference_dense_736_layer_call_and_return_conditional_losses_733819

inputs0
matmul_readvariableop_resource:Sa-
biasadd_readvariableop_resource:a
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Sa*
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
:ÿÿÿÿÿÿÿÿÿS: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_671_layer_call_and_return_conditional_losses_736892

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
Ð
²
S__inference_batch_normalization_672_layer_call_and_return_conditional_losses_736957

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
â
B
__inference__traced_save_737577
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	/
+savev2_dense_734_kernel_read_readvariableop-
)savev2_dense_734_bias_read_readvariableop<
8savev2_batch_normalization_665_gamma_read_readvariableop;
7savev2_batch_normalization_665_beta_read_readvariableopB
>savev2_batch_normalization_665_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_665_moving_variance_read_readvariableop/
+savev2_dense_735_kernel_read_readvariableop-
)savev2_dense_735_bias_read_readvariableop<
8savev2_batch_normalization_666_gamma_read_readvariableop;
7savev2_batch_normalization_666_beta_read_readvariableopB
>savev2_batch_normalization_666_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_666_moving_variance_read_readvariableop/
+savev2_dense_736_kernel_read_readvariableop-
)savev2_dense_736_bias_read_readvariableop<
8savev2_batch_normalization_667_gamma_read_readvariableop;
7savev2_batch_normalization_667_beta_read_readvariableopB
>savev2_batch_normalization_667_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_667_moving_variance_read_readvariableop/
+savev2_dense_737_kernel_read_readvariableop-
)savev2_dense_737_bias_read_readvariableop<
8savev2_batch_normalization_668_gamma_read_readvariableop;
7savev2_batch_normalization_668_beta_read_readvariableopB
>savev2_batch_normalization_668_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_668_moving_variance_read_readvariableop/
+savev2_dense_738_kernel_read_readvariableop-
)savev2_dense_738_bias_read_readvariableop<
8savev2_batch_normalization_669_gamma_read_readvariableop;
7savev2_batch_normalization_669_beta_read_readvariableopB
>savev2_batch_normalization_669_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_669_moving_variance_read_readvariableop/
+savev2_dense_739_kernel_read_readvariableop-
)savev2_dense_739_bias_read_readvariableop<
8savev2_batch_normalization_670_gamma_read_readvariableop;
7savev2_batch_normalization_670_beta_read_readvariableopB
>savev2_batch_normalization_670_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_670_moving_variance_read_readvariableop/
+savev2_dense_740_kernel_read_readvariableop-
)savev2_dense_740_bias_read_readvariableop<
8savev2_batch_normalization_671_gamma_read_readvariableop;
7savev2_batch_normalization_671_beta_read_readvariableopB
>savev2_batch_normalization_671_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_671_moving_variance_read_readvariableop/
+savev2_dense_741_kernel_read_readvariableop-
)savev2_dense_741_bias_read_readvariableop<
8savev2_batch_normalization_672_gamma_read_readvariableop;
7savev2_batch_normalization_672_beta_read_readvariableopB
>savev2_batch_normalization_672_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_672_moving_variance_read_readvariableop/
+savev2_dense_742_kernel_read_readvariableop-
)savev2_dense_742_bias_read_readvariableop<
8savev2_batch_normalization_673_gamma_read_readvariableop;
7savev2_batch_normalization_673_beta_read_readvariableopB
>savev2_batch_normalization_673_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_673_moving_variance_read_readvariableop/
+savev2_dense_743_kernel_read_readvariableop-
)savev2_dense_743_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_734_kernel_m_read_readvariableop4
0savev2_adam_dense_734_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_665_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_665_beta_m_read_readvariableop6
2savev2_adam_dense_735_kernel_m_read_readvariableop4
0savev2_adam_dense_735_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_666_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_666_beta_m_read_readvariableop6
2savev2_adam_dense_736_kernel_m_read_readvariableop4
0savev2_adam_dense_736_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_667_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_667_beta_m_read_readvariableop6
2savev2_adam_dense_737_kernel_m_read_readvariableop4
0savev2_adam_dense_737_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_668_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_668_beta_m_read_readvariableop6
2savev2_adam_dense_738_kernel_m_read_readvariableop4
0savev2_adam_dense_738_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_669_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_669_beta_m_read_readvariableop6
2savev2_adam_dense_739_kernel_m_read_readvariableop4
0savev2_adam_dense_739_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_670_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_670_beta_m_read_readvariableop6
2savev2_adam_dense_740_kernel_m_read_readvariableop4
0savev2_adam_dense_740_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_671_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_671_beta_m_read_readvariableop6
2savev2_adam_dense_741_kernel_m_read_readvariableop4
0savev2_adam_dense_741_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_672_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_672_beta_m_read_readvariableop6
2savev2_adam_dense_742_kernel_m_read_readvariableop4
0savev2_adam_dense_742_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_673_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_673_beta_m_read_readvariableop6
2savev2_adam_dense_743_kernel_m_read_readvariableop4
0savev2_adam_dense_743_bias_m_read_readvariableop6
2savev2_adam_dense_734_kernel_v_read_readvariableop4
0savev2_adam_dense_734_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_665_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_665_beta_v_read_readvariableop6
2savev2_adam_dense_735_kernel_v_read_readvariableop4
0savev2_adam_dense_735_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_666_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_666_beta_v_read_readvariableop6
2savev2_adam_dense_736_kernel_v_read_readvariableop4
0savev2_adam_dense_736_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_667_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_667_beta_v_read_readvariableop6
2savev2_adam_dense_737_kernel_v_read_readvariableop4
0savev2_adam_dense_737_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_668_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_668_beta_v_read_readvariableop6
2savev2_adam_dense_738_kernel_v_read_readvariableop4
0savev2_adam_dense_738_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_669_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_669_beta_v_read_readvariableop6
2savev2_adam_dense_739_kernel_v_read_readvariableop4
0savev2_adam_dense_739_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_670_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_670_beta_v_read_readvariableop6
2savev2_adam_dense_740_kernel_v_read_readvariableop4
0savev2_adam_dense_740_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_671_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_671_beta_v_read_readvariableop6
2savev2_adam_dense_741_kernel_v_read_readvariableop4
0savev2_adam_dense_741_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_672_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_672_beta_v_read_readvariableop6
2savev2_adam_dense_742_kernel_v_read_readvariableop4
0savev2_adam_dense_742_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_673_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_673_beta_v_read_readvariableop6
2savev2_adam_dense_743_kernel_v_read_readvariableop4
0savev2_adam_dense_743_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop+savev2_dense_734_kernel_read_readvariableop)savev2_dense_734_bias_read_readvariableop8savev2_batch_normalization_665_gamma_read_readvariableop7savev2_batch_normalization_665_beta_read_readvariableop>savev2_batch_normalization_665_moving_mean_read_readvariableopBsavev2_batch_normalization_665_moving_variance_read_readvariableop+savev2_dense_735_kernel_read_readvariableop)savev2_dense_735_bias_read_readvariableop8savev2_batch_normalization_666_gamma_read_readvariableop7savev2_batch_normalization_666_beta_read_readvariableop>savev2_batch_normalization_666_moving_mean_read_readvariableopBsavev2_batch_normalization_666_moving_variance_read_readvariableop+savev2_dense_736_kernel_read_readvariableop)savev2_dense_736_bias_read_readvariableop8savev2_batch_normalization_667_gamma_read_readvariableop7savev2_batch_normalization_667_beta_read_readvariableop>savev2_batch_normalization_667_moving_mean_read_readvariableopBsavev2_batch_normalization_667_moving_variance_read_readvariableop+savev2_dense_737_kernel_read_readvariableop)savev2_dense_737_bias_read_readvariableop8savev2_batch_normalization_668_gamma_read_readvariableop7savev2_batch_normalization_668_beta_read_readvariableop>savev2_batch_normalization_668_moving_mean_read_readvariableopBsavev2_batch_normalization_668_moving_variance_read_readvariableop+savev2_dense_738_kernel_read_readvariableop)savev2_dense_738_bias_read_readvariableop8savev2_batch_normalization_669_gamma_read_readvariableop7savev2_batch_normalization_669_beta_read_readvariableop>savev2_batch_normalization_669_moving_mean_read_readvariableopBsavev2_batch_normalization_669_moving_variance_read_readvariableop+savev2_dense_739_kernel_read_readvariableop)savev2_dense_739_bias_read_readvariableop8savev2_batch_normalization_670_gamma_read_readvariableop7savev2_batch_normalization_670_beta_read_readvariableop>savev2_batch_normalization_670_moving_mean_read_readvariableopBsavev2_batch_normalization_670_moving_variance_read_readvariableop+savev2_dense_740_kernel_read_readvariableop)savev2_dense_740_bias_read_readvariableop8savev2_batch_normalization_671_gamma_read_readvariableop7savev2_batch_normalization_671_beta_read_readvariableop>savev2_batch_normalization_671_moving_mean_read_readvariableopBsavev2_batch_normalization_671_moving_variance_read_readvariableop+savev2_dense_741_kernel_read_readvariableop)savev2_dense_741_bias_read_readvariableop8savev2_batch_normalization_672_gamma_read_readvariableop7savev2_batch_normalization_672_beta_read_readvariableop>savev2_batch_normalization_672_moving_mean_read_readvariableopBsavev2_batch_normalization_672_moving_variance_read_readvariableop+savev2_dense_742_kernel_read_readvariableop)savev2_dense_742_bias_read_readvariableop8savev2_batch_normalization_673_gamma_read_readvariableop7savev2_batch_normalization_673_beta_read_readvariableop>savev2_batch_normalization_673_moving_mean_read_readvariableopBsavev2_batch_normalization_673_moving_variance_read_readvariableop+savev2_dense_743_kernel_read_readvariableop)savev2_dense_743_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_734_kernel_m_read_readvariableop0savev2_adam_dense_734_bias_m_read_readvariableop?savev2_adam_batch_normalization_665_gamma_m_read_readvariableop>savev2_adam_batch_normalization_665_beta_m_read_readvariableop2savev2_adam_dense_735_kernel_m_read_readvariableop0savev2_adam_dense_735_bias_m_read_readvariableop?savev2_adam_batch_normalization_666_gamma_m_read_readvariableop>savev2_adam_batch_normalization_666_beta_m_read_readvariableop2savev2_adam_dense_736_kernel_m_read_readvariableop0savev2_adam_dense_736_bias_m_read_readvariableop?savev2_adam_batch_normalization_667_gamma_m_read_readvariableop>savev2_adam_batch_normalization_667_beta_m_read_readvariableop2savev2_adam_dense_737_kernel_m_read_readvariableop0savev2_adam_dense_737_bias_m_read_readvariableop?savev2_adam_batch_normalization_668_gamma_m_read_readvariableop>savev2_adam_batch_normalization_668_beta_m_read_readvariableop2savev2_adam_dense_738_kernel_m_read_readvariableop0savev2_adam_dense_738_bias_m_read_readvariableop?savev2_adam_batch_normalization_669_gamma_m_read_readvariableop>savev2_adam_batch_normalization_669_beta_m_read_readvariableop2savev2_adam_dense_739_kernel_m_read_readvariableop0savev2_adam_dense_739_bias_m_read_readvariableop?savev2_adam_batch_normalization_670_gamma_m_read_readvariableop>savev2_adam_batch_normalization_670_beta_m_read_readvariableop2savev2_adam_dense_740_kernel_m_read_readvariableop0savev2_adam_dense_740_bias_m_read_readvariableop?savev2_adam_batch_normalization_671_gamma_m_read_readvariableop>savev2_adam_batch_normalization_671_beta_m_read_readvariableop2savev2_adam_dense_741_kernel_m_read_readvariableop0savev2_adam_dense_741_bias_m_read_readvariableop?savev2_adam_batch_normalization_672_gamma_m_read_readvariableop>savev2_adam_batch_normalization_672_beta_m_read_readvariableop2savev2_adam_dense_742_kernel_m_read_readvariableop0savev2_adam_dense_742_bias_m_read_readvariableop?savev2_adam_batch_normalization_673_gamma_m_read_readvariableop>savev2_adam_batch_normalization_673_beta_m_read_readvariableop2savev2_adam_dense_743_kernel_m_read_readvariableop0savev2_adam_dense_743_bias_m_read_readvariableop2savev2_adam_dense_734_kernel_v_read_readvariableop0savev2_adam_dense_734_bias_v_read_readvariableop?savev2_adam_batch_normalization_665_gamma_v_read_readvariableop>savev2_adam_batch_normalization_665_beta_v_read_readvariableop2savev2_adam_dense_735_kernel_v_read_readvariableop0savev2_adam_dense_735_bias_v_read_readvariableop?savev2_adam_batch_normalization_666_gamma_v_read_readvariableop>savev2_adam_batch_normalization_666_beta_v_read_readvariableop2savev2_adam_dense_736_kernel_v_read_readvariableop0savev2_adam_dense_736_bias_v_read_readvariableop?savev2_adam_batch_normalization_667_gamma_v_read_readvariableop>savev2_adam_batch_normalization_667_beta_v_read_readvariableop2savev2_adam_dense_737_kernel_v_read_readvariableop0savev2_adam_dense_737_bias_v_read_readvariableop?savev2_adam_batch_normalization_668_gamma_v_read_readvariableop>savev2_adam_batch_normalization_668_beta_v_read_readvariableop2savev2_adam_dense_738_kernel_v_read_readvariableop0savev2_adam_dense_738_bias_v_read_readvariableop?savev2_adam_batch_normalization_669_gamma_v_read_readvariableop>savev2_adam_batch_normalization_669_beta_v_read_readvariableop2savev2_adam_dense_739_kernel_v_read_readvariableop0savev2_adam_dense_739_bias_v_read_readvariableop?savev2_adam_batch_normalization_670_gamma_v_read_readvariableop>savev2_adam_batch_normalization_670_beta_v_read_readvariableop2savev2_adam_dense_740_kernel_v_read_readvariableop0savev2_adam_dense_740_bias_v_read_readvariableop?savev2_adam_batch_normalization_671_gamma_v_read_readvariableop>savev2_adam_batch_normalization_671_beta_v_read_readvariableop2savev2_adam_dense_741_kernel_v_read_readvariableop0savev2_adam_dense_741_bias_v_read_readvariableop?savev2_adam_batch_normalization_672_gamma_v_read_readvariableop>savev2_adam_batch_normalization_672_beta_v_read_readvariableop2savev2_adam_dense_742_kernel_v_read_readvariableop0savev2_adam_dense_742_bias_v_read_readvariableop?savev2_adam_batch_normalization_673_gamma_v_read_readvariableop>savev2_adam_batch_normalization_673_beta_v_read_readvariableop2savev2_adam_dense_743_kernel_v_read_readvariableop0savev2_adam_dense_743_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
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
®: ::: :S:S:S:S:S:S:SS:S:S:S:S:S:Sa:a:a:a:a:a:aa:a:a:a:a:a:aA:A:A:A:A:A:AA:A:A:A:A:A:AA:A:A:A:A:A:AA:A:A:A:A:A:AA:A:A:A:A:A:A:: : : : : : :S:S:S:S:SS:S:S:S:Sa:a:a:a:aa:a:a:a:aA:A:A:A:AA:A:A:A:AA:A:A:A:AA:A:A:A:AA:A:A:A:A::S:S:S:S:SS:S:S:S:Sa:a:a:a:aa:a:a:a:aA:A:A:A:AA:A:A:A:AA:A:A:A:AA:A:A:A:AA:A:A:A:A:: 2(
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

:S: 

_output_shapes
:S: 

_output_shapes
:S: 

_output_shapes
:S: 

_output_shapes
:S: 	

_output_shapes
:S:$
 

_output_shapes

:SS: 

_output_shapes
:S: 

_output_shapes
:S: 

_output_shapes
:S: 

_output_shapes
:S: 

_output_shapes
:S:$ 

_output_shapes

:Sa: 
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

:aA: 

_output_shapes
:A: 

_output_shapes
:A: 

_output_shapes
:A:  

_output_shapes
:A: !

_output_shapes
:A:$" 

_output_shapes

:AA: #

_output_shapes
:A: $

_output_shapes
:A: %

_output_shapes
:A: &

_output_shapes
:A: '

_output_shapes
:A:$( 

_output_shapes

:AA: )

_output_shapes
:A: *

_output_shapes
:A: +

_output_shapes
:A: ,

_output_shapes
:A: -

_output_shapes
:A:$. 

_output_shapes

:AA: /

_output_shapes
:A: 0

_output_shapes
:A: 1

_output_shapes
:A: 2

_output_shapes
:A: 3

_output_shapes
:A:$4 

_output_shapes

:AA: 5

_output_shapes
:A: 6

_output_shapes
:A: 7

_output_shapes
:A: 8

_output_shapes
:A: 9

_output_shapes
:A:$: 

_output_shapes

:A: ;
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

:S: C

_output_shapes
:S: D

_output_shapes
:S: E

_output_shapes
:S:$F 

_output_shapes

:SS: G

_output_shapes
:S: H

_output_shapes
:S: I

_output_shapes
:S:$J 

_output_shapes

:Sa: K
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

:aA: S

_output_shapes
:A: T

_output_shapes
:A: U

_output_shapes
:A:$V 

_output_shapes

:AA: W

_output_shapes
:A: X

_output_shapes
:A: Y

_output_shapes
:A:$Z 

_output_shapes

:AA: [

_output_shapes
:A: \

_output_shapes
:A: ]

_output_shapes
:A:$^ 

_output_shapes

:AA: _

_output_shapes
:A: `

_output_shapes
:A: a

_output_shapes
:A:$b 

_output_shapes

:AA: c

_output_shapes
:A: d

_output_shapes
:A: e

_output_shapes
:A:$f 

_output_shapes

:A: g

_output_shapes
::$h 

_output_shapes

:S: i

_output_shapes
:S: j

_output_shapes
:S: k

_output_shapes
:S:$l 

_output_shapes

:SS: m

_output_shapes
:S: n

_output_shapes
:S: o

_output_shapes
:S:$p 

_output_shapes

:Sa: q
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

:aA: y

_output_shapes
:A: z

_output_shapes
:A: {

_output_shapes
:A:$| 

_output_shapes

:AA: }

_output_shapes
:A: ~

_output_shapes
:A: 

_output_shapes
:A:% 

_output_shapes

:AA:!

_output_shapes
:A:!

_output_shapes
:A:!

_output_shapes
:A:% 

_output_shapes

:AA:!

_output_shapes
:A:!

_output_shapes
:A:!

_output_shapes
:A:% 

_output_shapes

:AA:!

_output_shapes
:A:!

_output_shapes
:A:!

_output_shapes
:A:% 

_output_shapes

:A:!

_output_shapes
::

_output_shapes
: 
Ð
²
S__inference_batch_normalization_665_layer_call_and_return_conditional_losses_733017

inputs/
!batchnorm_readvariableop_resource:S3
%batchnorm_mul_readvariableop_resource:S1
#batchnorm_readvariableop_1_resource:S1
#batchnorm_readvariableop_2_resource:S
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:S*
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
:SP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:S~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:S*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Sc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:S*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Sz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:S*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Sr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿS: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
È	
ö
E__inference_dense_738_layer_call_and_return_conditional_losses_733883

inputs0
matmul_readvariableop_resource:aA-
biasadd_readvariableop_resource:A
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:aA*
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
:ÿÿÿÿÿÿÿÿÿa: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 
_user_specified_nameinputs
õ
â
.__inference_sequential_69_layer_call_fn_734837
normalization_69_input
unknown
	unknown_0
	unknown_1:S
	unknown_2:S
	unknown_3:S
	unknown_4:S
	unknown_5:S
	unknown_6:S
	unknown_7:SS
	unknown_8:S
	unknown_9:S

unknown_10:S

unknown_11:S

unknown_12:S

unknown_13:Sa

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

unknown_25:aA

unknown_26:A

unknown_27:A

unknown_28:A

unknown_29:A

unknown_30:A

unknown_31:AA

unknown_32:A

unknown_33:A

unknown_34:A

unknown_35:A

unknown_36:A

unknown_37:AA

unknown_38:A

unknown_39:A

unknown_40:A

unknown_41:A

unknown_42:A

unknown_43:AA

unknown_44:A

unknown_45:A

unknown_46:A

unknown_47:A

unknown_48:A

unknown_49:AA

unknown_50:A

unknown_51:A

unknown_52:A

unknown_53:A

unknown_54:A

unknown_55:A

unknown_56:
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallnormalization_69_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_69_layer_call_and_return_conditional_losses_734597o
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
_user_specified_namenormalization_69_input:$ 

_output_shapes

::$ 

_output_shapes

:
È	
ö
E__inference_dense_739_layer_call_and_return_conditional_losses_733915

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
%
ì
S__inference_batch_normalization_671_layer_call_and_return_conditional_losses_736882

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
å
g
K__inference_leaky_re_lu_667_layer_call_and_return_conditional_losses_733839

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
Ð
²
S__inference_batch_normalization_673_layer_call_and_return_conditional_losses_733673

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
å
g
K__inference_leaky_re_lu_669_layer_call_and_return_conditional_losses_733903

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
%
ì
S__inference_batch_normalization_669_layer_call_and_return_conditional_losses_733392

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
%
ì
S__inference_batch_normalization_666_layer_call_and_return_conditional_losses_733146

inputs5
'assignmovingavg_readvariableop_resource:S7
)assignmovingavg_1_readvariableop_resource:S3
%batchnorm_mul_readvariableop_resource:S/
!batchnorm_readvariableop_resource:S
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:S*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:S
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:S*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:S*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:S*
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
:S*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Sx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:S¬
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
:S*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:S~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:S´
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
:SP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:S~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:S*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Sc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Sv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:S*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Sr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿS: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_665_layer_call_and_return_conditional_losses_736194

inputs/
!batchnorm_readvariableop_resource:S3
%batchnorm_mul_readvariableop_resource:S1
#batchnorm_readvariableop_1_resource:S1
#batchnorm_readvariableop_2_resource:S
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:S*
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
:SP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:S~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:S*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Sc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:S*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Sz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:S*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Sr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿS: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_670_layer_call_and_return_conditional_losses_736783

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
%
ì
S__inference_batch_normalization_671_layer_call_and_return_conditional_losses_733556

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
Õ
Ø
$__inference_signature_wrapper_736082
normalization_69_input
unknown
	unknown_0
	unknown_1:S
	unknown_2:S
	unknown_3:S
	unknown_4:S
	unknown_5:S
	unknown_6:S
	unknown_7:SS
	unknown_8:S
	unknown_9:S

unknown_10:S

unknown_11:S

unknown_12:S

unknown_13:Sa

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

unknown_25:aA

unknown_26:A

unknown_27:A

unknown_28:A

unknown_29:A

unknown_30:A

unknown_31:AA

unknown_32:A

unknown_33:A

unknown_34:A

unknown_35:A

unknown_36:A

unknown_37:AA

unknown_38:A

unknown_39:A

unknown_40:A

unknown_41:A

unknown_42:A

unknown_43:AA

unknown_44:A

unknown_45:A

unknown_46:A

unknown_47:A

unknown_48:A

unknown_49:AA

unknown_50:A

unknown_51:A

unknown_52:A

unknown_53:A

unknown_54:A

unknown_55:A

unknown_56:
identity¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallnormalization_69_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_732993o
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
_user_specified_namenormalization_69_input:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_673_layer_call_and_return_conditional_losses_734031

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
ª
Ó
8__inference_batch_normalization_671_layer_call_fn_736828

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_671_layer_call_and_return_conditional_losses_733556o
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
å
g
K__inference_leaky_re_lu_666_layer_call_and_return_conditional_losses_736347

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿS:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_673_layer_call_and_return_conditional_losses_733720

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
Ð
²
S__inference_batch_normalization_667_layer_call_and_return_conditional_losses_736412

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
8__inference_batch_normalization_669_layer_call_fn_736597

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_669_layer_call_and_return_conditional_losses_733345o
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
¥
ë
I__inference_sequential_69_layer_call_and_return_conditional_losses_734050

inputs
normalization_69_sub_y
normalization_69_sqrt_x"
dense_734_733756:S
dense_734_733758:S,
batch_normalization_665_733761:S,
batch_normalization_665_733763:S,
batch_normalization_665_733765:S,
batch_normalization_665_733767:S"
dense_735_733788:SS
dense_735_733790:S,
batch_normalization_666_733793:S,
batch_normalization_666_733795:S,
batch_normalization_666_733797:S,
batch_normalization_666_733799:S"
dense_736_733820:Sa
dense_736_733822:a,
batch_normalization_667_733825:a,
batch_normalization_667_733827:a,
batch_normalization_667_733829:a,
batch_normalization_667_733831:a"
dense_737_733852:aa
dense_737_733854:a,
batch_normalization_668_733857:a,
batch_normalization_668_733859:a,
batch_normalization_668_733861:a,
batch_normalization_668_733863:a"
dense_738_733884:aA
dense_738_733886:A,
batch_normalization_669_733889:A,
batch_normalization_669_733891:A,
batch_normalization_669_733893:A,
batch_normalization_669_733895:A"
dense_739_733916:AA
dense_739_733918:A,
batch_normalization_670_733921:A,
batch_normalization_670_733923:A,
batch_normalization_670_733925:A,
batch_normalization_670_733927:A"
dense_740_733948:AA
dense_740_733950:A,
batch_normalization_671_733953:A,
batch_normalization_671_733955:A,
batch_normalization_671_733957:A,
batch_normalization_671_733959:A"
dense_741_733980:AA
dense_741_733982:A,
batch_normalization_672_733985:A,
batch_normalization_672_733987:A,
batch_normalization_672_733989:A,
batch_normalization_672_733991:A"
dense_742_734012:AA
dense_742_734014:A,
batch_normalization_673_734017:A,
batch_normalization_673_734019:A,
batch_normalization_673_734021:A,
batch_normalization_673_734023:A"
dense_743_734044:A
dense_743_734046:
identity¢/batch_normalization_665/StatefulPartitionedCall¢/batch_normalization_666/StatefulPartitionedCall¢/batch_normalization_667/StatefulPartitionedCall¢/batch_normalization_668/StatefulPartitionedCall¢/batch_normalization_669/StatefulPartitionedCall¢/batch_normalization_670/StatefulPartitionedCall¢/batch_normalization_671/StatefulPartitionedCall¢/batch_normalization_672/StatefulPartitionedCall¢/batch_normalization_673/StatefulPartitionedCall¢!dense_734/StatefulPartitionedCall¢!dense_735/StatefulPartitionedCall¢!dense_736/StatefulPartitionedCall¢!dense_737/StatefulPartitionedCall¢!dense_738/StatefulPartitionedCall¢!dense_739/StatefulPartitionedCall¢!dense_740/StatefulPartitionedCall¢!dense_741/StatefulPartitionedCall¢!dense_742/StatefulPartitionedCall¢!dense_743/StatefulPartitionedCallm
normalization_69/subSubinputsnormalization_69_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_69/SqrtSqrtnormalization_69_sqrt_x*
T0*
_output_shapes

:_
normalization_69/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_69/MaximumMaximumnormalization_69/Sqrt:y:0#normalization_69/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_69/truedivRealDivnormalization_69/sub:z:0normalization_69/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_734/StatefulPartitionedCallStatefulPartitionedCallnormalization_69/truediv:z:0dense_734_733756dense_734_733758*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_734_layer_call_and_return_conditional_losses_733755
/batch_normalization_665/StatefulPartitionedCallStatefulPartitionedCall*dense_734/StatefulPartitionedCall:output:0batch_normalization_665_733761batch_normalization_665_733763batch_normalization_665_733765batch_normalization_665_733767*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_665_layer_call_and_return_conditional_losses_733017ø
leaky_re_lu_665/PartitionedCallPartitionedCall8batch_normalization_665/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_665_layer_call_and_return_conditional_losses_733775
!dense_735/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_665/PartitionedCall:output:0dense_735_733788dense_735_733790*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_735_layer_call_and_return_conditional_losses_733787
/batch_normalization_666/StatefulPartitionedCallStatefulPartitionedCall*dense_735/StatefulPartitionedCall:output:0batch_normalization_666_733793batch_normalization_666_733795batch_normalization_666_733797batch_normalization_666_733799*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_666_layer_call_and_return_conditional_losses_733099ø
leaky_re_lu_666/PartitionedCallPartitionedCall8batch_normalization_666/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_666_layer_call_and_return_conditional_losses_733807
!dense_736/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_666/PartitionedCall:output:0dense_736_733820dense_736_733822*
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
E__inference_dense_736_layer_call_and_return_conditional_losses_733819
/batch_normalization_667/StatefulPartitionedCallStatefulPartitionedCall*dense_736/StatefulPartitionedCall:output:0batch_normalization_667_733825batch_normalization_667_733827batch_normalization_667_733829batch_normalization_667_733831*
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
S__inference_batch_normalization_667_layer_call_and_return_conditional_losses_733181ø
leaky_re_lu_667/PartitionedCallPartitionedCall8batch_normalization_667/StatefulPartitionedCall:output:0*
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
K__inference_leaky_re_lu_667_layer_call_and_return_conditional_losses_733839
!dense_737/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_667/PartitionedCall:output:0dense_737_733852dense_737_733854*
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
E__inference_dense_737_layer_call_and_return_conditional_losses_733851
/batch_normalization_668/StatefulPartitionedCallStatefulPartitionedCall*dense_737/StatefulPartitionedCall:output:0batch_normalization_668_733857batch_normalization_668_733859batch_normalization_668_733861batch_normalization_668_733863*
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
S__inference_batch_normalization_668_layer_call_and_return_conditional_losses_733263ø
leaky_re_lu_668/PartitionedCallPartitionedCall8batch_normalization_668/StatefulPartitionedCall:output:0*
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
K__inference_leaky_re_lu_668_layer_call_and_return_conditional_losses_733871
!dense_738/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_668/PartitionedCall:output:0dense_738_733884dense_738_733886*
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
GPU 2J 8 *N
fIRG
E__inference_dense_738_layer_call_and_return_conditional_losses_733883
/batch_normalization_669/StatefulPartitionedCallStatefulPartitionedCall*dense_738/StatefulPartitionedCall:output:0batch_normalization_669_733889batch_normalization_669_733891batch_normalization_669_733893batch_normalization_669_733895*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_669_layer_call_and_return_conditional_losses_733345ø
leaky_re_lu_669/PartitionedCallPartitionedCall8batch_normalization_669/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_669_layer_call_and_return_conditional_losses_733903
!dense_739/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_669/PartitionedCall:output:0dense_739_733916dense_739_733918*
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
GPU 2J 8 *N
fIRG
E__inference_dense_739_layer_call_and_return_conditional_losses_733915
/batch_normalization_670/StatefulPartitionedCallStatefulPartitionedCall*dense_739/StatefulPartitionedCall:output:0batch_normalization_670_733921batch_normalization_670_733923batch_normalization_670_733925batch_normalization_670_733927*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_670_layer_call_and_return_conditional_losses_733427ø
leaky_re_lu_670/PartitionedCallPartitionedCall8batch_normalization_670/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_670_layer_call_and_return_conditional_losses_733935
!dense_740/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_670/PartitionedCall:output:0dense_740_733948dense_740_733950*
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
GPU 2J 8 *N
fIRG
E__inference_dense_740_layer_call_and_return_conditional_losses_733947
/batch_normalization_671/StatefulPartitionedCallStatefulPartitionedCall*dense_740/StatefulPartitionedCall:output:0batch_normalization_671_733953batch_normalization_671_733955batch_normalization_671_733957batch_normalization_671_733959*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_671_layer_call_and_return_conditional_losses_733509ø
leaky_re_lu_671/PartitionedCallPartitionedCall8batch_normalization_671/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_671_layer_call_and_return_conditional_losses_733967
!dense_741/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_671/PartitionedCall:output:0dense_741_733980dense_741_733982*
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
GPU 2J 8 *N
fIRG
E__inference_dense_741_layer_call_and_return_conditional_losses_733979
/batch_normalization_672/StatefulPartitionedCallStatefulPartitionedCall*dense_741/StatefulPartitionedCall:output:0batch_normalization_672_733985batch_normalization_672_733987batch_normalization_672_733989batch_normalization_672_733991*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_672_layer_call_and_return_conditional_losses_733591ø
leaky_re_lu_672/PartitionedCallPartitionedCall8batch_normalization_672/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_672_layer_call_and_return_conditional_losses_733999
!dense_742/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_672/PartitionedCall:output:0dense_742_734012dense_742_734014*
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
GPU 2J 8 *N
fIRG
E__inference_dense_742_layer_call_and_return_conditional_losses_734011
/batch_normalization_673/StatefulPartitionedCallStatefulPartitionedCall*dense_742/StatefulPartitionedCall:output:0batch_normalization_673_734017batch_normalization_673_734019batch_normalization_673_734021batch_normalization_673_734023*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_673_layer_call_and_return_conditional_losses_733673ø
leaky_re_lu_673/PartitionedCallPartitionedCall8batch_normalization_673/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_673_layer_call_and_return_conditional_losses_734031
!dense_743/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_673/PartitionedCall:output:0dense_743_734044dense_743_734046*
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
E__inference_dense_743_layer_call_and_return_conditional_losses_734043y
IdentityIdentity*dense_743/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp0^batch_normalization_665/StatefulPartitionedCall0^batch_normalization_666/StatefulPartitionedCall0^batch_normalization_667/StatefulPartitionedCall0^batch_normalization_668/StatefulPartitionedCall0^batch_normalization_669/StatefulPartitionedCall0^batch_normalization_670/StatefulPartitionedCall0^batch_normalization_671/StatefulPartitionedCall0^batch_normalization_672/StatefulPartitionedCall0^batch_normalization_673/StatefulPartitionedCall"^dense_734/StatefulPartitionedCall"^dense_735/StatefulPartitionedCall"^dense_736/StatefulPartitionedCall"^dense_737/StatefulPartitionedCall"^dense_738/StatefulPartitionedCall"^dense_739/StatefulPartitionedCall"^dense_740/StatefulPartitionedCall"^dense_741/StatefulPartitionedCall"^dense_742/StatefulPartitionedCall"^dense_743/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_665/StatefulPartitionedCall/batch_normalization_665/StatefulPartitionedCall2b
/batch_normalization_666/StatefulPartitionedCall/batch_normalization_666/StatefulPartitionedCall2b
/batch_normalization_667/StatefulPartitionedCall/batch_normalization_667/StatefulPartitionedCall2b
/batch_normalization_668/StatefulPartitionedCall/batch_normalization_668/StatefulPartitionedCall2b
/batch_normalization_669/StatefulPartitionedCall/batch_normalization_669/StatefulPartitionedCall2b
/batch_normalization_670/StatefulPartitionedCall/batch_normalization_670/StatefulPartitionedCall2b
/batch_normalization_671/StatefulPartitionedCall/batch_normalization_671/StatefulPartitionedCall2b
/batch_normalization_672/StatefulPartitionedCall/batch_normalization_672/StatefulPartitionedCall2b
/batch_normalization_673/StatefulPartitionedCall/batch_normalization_673/StatefulPartitionedCall2F
!dense_734/StatefulPartitionedCall!dense_734/StatefulPartitionedCall2F
!dense_735/StatefulPartitionedCall!dense_735/StatefulPartitionedCall2F
!dense_736/StatefulPartitionedCall!dense_736/StatefulPartitionedCall2F
!dense_737/StatefulPartitionedCall!dense_737/StatefulPartitionedCall2F
!dense_738/StatefulPartitionedCall!dense_738/StatefulPartitionedCall2F
!dense_739/StatefulPartitionedCall!dense_739/StatefulPartitionedCall2F
!dense_740/StatefulPartitionedCall!dense_740/StatefulPartitionedCall2F
!dense_741/StatefulPartitionedCall!dense_741/StatefulPartitionedCall2F
!dense_742/StatefulPartitionedCall!dense_742/StatefulPartitionedCall2F
!dense_743/StatefulPartitionedCall!dense_743/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
¦Â
å3
I__inference_sequential_69_layer_call_and_return_conditional_losses_735609

inputs
normalization_69_sub_y
normalization_69_sqrt_x:
(dense_734_matmul_readvariableop_resource:S7
)dense_734_biasadd_readvariableop_resource:SG
9batch_normalization_665_batchnorm_readvariableop_resource:SK
=batch_normalization_665_batchnorm_mul_readvariableop_resource:SI
;batch_normalization_665_batchnorm_readvariableop_1_resource:SI
;batch_normalization_665_batchnorm_readvariableop_2_resource:S:
(dense_735_matmul_readvariableop_resource:SS7
)dense_735_biasadd_readvariableop_resource:SG
9batch_normalization_666_batchnorm_readvariableop_resource:SK
=batch_normalization_666_batchnorm_mul_readvariableop_resource:SI
;batch_normalization_666_batchnorm_readvariableop_1_resource:SI
;batch_normalization_666_batchnorm_readvariableop_2_resource:S:
(dense_736_matmul_readvariableop_resource:Sa7
)dense_736_biasadd_readvariableop_resource:aG
9batch_normalization_667_batchnorm_readvariableop_resource:aK
=batch_normalization_667_batchnorm_mul_readvariableop_resource:aI
;batch_normalization_667_batchnorm_readvariableop_1_resource:aI
;batch_normalization_667_batchnorm_readvariableop_2_resource:a:
(dense_737_matmul_readvariableop_resource:aa7
)dense_737_biasadd_readvariableop_resource:aG
9batch_normalization_668_batchnorm_readvariableop_resource:aK
=batch_normalization_668_batchnorm_mul_readvariableop_resource:aI
;batch_normalization_668_batchnorm_readvariableop_1_resource:aI
;batch_normalization_668_batchnorm_readvariableop_2_resource:a:
(dense_738_matmul_readvariableop_resource:aA7
)dense_738_biasadd_readvariableop_resource:AG
9batch_normalization_669_batchnorm_readvariableop_resource:AK
=batch_normalization_669_batchnorm_mul_readvariableop_resource:AI
;batch_normalization_669_batchnorm_readvariableop_1_resource:AI
;batch_normalization_669_batchnorm_readvariableop_2_resource:A:
(dense_739_matmul_readvariableop_resource:AA7
)dense_739_biasadd_readvariableop_resource:AG
9batch_normalization_670_batchnorm_readvariableop_resource:AK
=batch_normalization_670_batchnorm_mul_readvariableop_resource:AI
;batch_normalization_670_batchnorm_readvariableop_1_resource:AI
;batch_normalization_670_batchnorm_readvariableop_2_resource:A:
(dense_740_matmul_readvariableop_resource:AA7
)dense_740_biasadd_readvariableop_resource:AG
9batch_normalization_671_batchnorm_readvariableop_resource:AK
=batch_normalization_671_batchnorm_mul_readvariableop_resource:AI
;batch_normalization_671_batchnorm_readvariableop_1_resource:AI
;batch_normalization_671_batchnorm_readvariableop_2_resource:A:
(dense_741_matmul_readvariableop_resource:AA7
)dense_741_biasadd_readvariableop_resource:AG
9batch_normalization_672_batchnorm_readvariableop_resource:AK
=batch_normalization_672_batchnorm_mul_readvariableop_resource:AI
;batch_normalization_672_batchnorm_readvariableop_1_resource:AI
;batch_normalization_672_batchnorm_readvariableop_2_resource:A:
(dense_742_matmul_readvariableop_resource:AA7
)dense_742_biasadd_readvariableop_resource:AG
9batch_normalization_673_batchnorm_readvariableop_resource:AK
=batch_normalization_673_batchnorm_mul_readvariableop_resource:AI
;batch_normalization_673_batchnorm_readvariableop_1_resource:AI
;batch_normalization_673_batchnorm_readvariableop_2_resource:A:
(dense_743_matmul_readvariableop_resource:A7
)dense_743_biasadd_readvariableop_resource:
identity¢0batch_normalization_665/batchnorm/ReadVariableOp¢2batch_normalization_665/batchnorm/ReadVariableOp_1¢2batch_normalization_665/batchnorm/ReadVariableOp_2¢4batch_normalization_665/batchnorm/mul/ReadVariableOp¢0batch_normalization_666/batchnorm/ReadVariableOp¢2batch_normalization_666/batchnorm/ReadVariableOp_1¢2batch_normalization_666/batchnorm/ReadVariableOp_2¢4batch_normalization_666/batchnorm/mul/ReadVariableOp¢0batch_normalization_667/batchnorm/ReadVariableOp¢2batch_normalization_667/batchnorm/ReadVariableOp_1¢2batch_normalization_667/batchnorm/ReadVariableOp_2¢4batch_normalization_667/batchnorm/mul/ReadVariableOp¢0batch_normalization_668/batchnorm/ReadVariableOp¢2batch_normalization_668/batchnorm/ReadVariableOp_1¢2batch_normalization_668/batchnorm/ReadVariableOp_2¢4batch_normalization_668/batchnorm/mul/ReadVariableOp¢0batch_normalization_669/batchnorm/ReadVariableOp¢2batch_normalization_669/batchnorm/ReadVariableOp_1¢2batch_normalization_669/batchnorm/ReadVariableOp_2¢4batch_normalization_669/batchnorm/mul/ReadVariableOp¢0batch_normalization_670/batchnorm/ReadVariableOp¢2batch_normalization_670/batchnorm/ReadVariableOp_1¢2batch_normalization_670/batchnorm/ReadVariableOp_2¢4batch_normalization_670/batchnorm/mul/ReadVariableOp¢0batch_normalization_671/batchnorm/ReadVariableOp¢2batch_normalization_671/batchnorm/ReadVariableOp_1¢2batch_normalization_671/batchnorm/ReadVariableOp_2¢4batch_normalization_671/batchnorm/mul/ReadVariableOp¢0batch_normalization_672/batchnorm/ReadVariableOp¢2batch_normalization_672/batchnorm/ReadVariableOp_1¢2batch_normalization_672/batchnorm/ReadVariableOp_2¢4batch_normalization_672/batchnorm/mul/ReadVariableOp¢0batch_normalization_673/batchnorm/ReadVariableOp¢2batch_normalization_673/batchnorm/ReadVariableOp_1¢2batch_normalization_673/batchnorm/ReadVariableOp_2¢4batch_normalization_673/batchnorm/mul/ReadVariableOp¢ dense_734/BiasAdd/ReadVariableOp¢dense_734/MatMul/ReadVariableOp¢ dense_735/BiasAdd/ReadVariableOp¢dense_735/MatMul/ReadVariableOp¢ dense_736/BiasAdd/ReadVariableOp¢dense_736/MatMul/ReadVariableOp¢ dense_737/BiasAdd/ReadVariableOp¢dense_737/MatMul/ReadVariableOp¢ dense_738/BiasAdd/ReadVariableOp¢dense_738/MatMul/ReadVariableOp¢ dense_739/BiasAdd/ReadVariableOp¢dense_739/MatMul/ReadVariableOp¢ dense_740/BiasAdd/ReadVariableOp¢dense_740/MatMul/ReadVariableOp¢ dense_741/BiasAdd/ReadVariableOp¢dense_741/MatMul/ReadVariableOp¢ dense_742/BiasAdd/ReadVariableOp¢dense_742/MatMul/ReadVariableOp¢ dense_743/BiasAdd/ReadVariableOp¢dense_743/MatMul/ReadVariableOpm
normalization_69/subSubinputsnormalization_69_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_69/SqrtSqrtnormalization_69_sqrt_x*
T0*
_output_shapes

:_
normalization_69/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_69/MaximumMaximumnormalization_69/Sqrt:y:0#normalization_69/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_69/truedivRealDivnormalization_69/sub:z:0normalization_69/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_734/MatMul/ReadVariableOpReadVariableOp(dense_734_matmul_readvariableop_resource*
_output_shapes

:S*
dtype0
dense_734/MatMulMatMulnormalization_69/truediv:z:0'dense_734/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 dense_734/BiasAdd/ReadVariableOpReadVariableOp)dense_734_biasadd_readvariableop_resource*
_output_shapes
:S*
dtype0
dense_734/BiasAddBiasAdddense_734/MatMul:product:0(dense_734/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS¦
0batch_normalization_665/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_665_batchnorm_readvariableop_resource*
_output_shapes
:S*
dtype0l
'batch_normalization_665/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_665/batchnorm/addAddV28batch_normalization_665/batchnorm/ReadVariableOp:value:00batch_normalization_665/batchnorm/add/y:output:0*
T0*
_output_shapes
:S
'batch_normalization_665/batchnorm/RsqrtRsqrt)batch_normalization_665/batchnorm/add:z:0*
T0*
_output_shapes
:S®
4batch_normalization_665/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_665_batchnorm_mul_readvariableop_resource*
_output_shapes
:S*
dtype0¼
%batch_normalization_665/batchnorm/mulMul+batch_normalization_665/batchnorm/Rsqrt:y:0<batch_normalization_665/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:S§
'batch_normalization_665/batchnorm/mul_1Muldense_734/BiasAdd:output:0)batch_normalization_665/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSª
2batch_normalization_665/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_665_batchnorm_readvariableop_1_resource*
_output_shapes
:S*
dtype0º
'batch_normalization_665/batchnorm/mul_2Mul:batch_normalization_665/batchnorm/ReadVariableOp_1:value:0)batch_normalization_665/batchnorm/mul:z:0*
T0*
_output_shapes
:Sª
2batch_normalization_665/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_665_batchnorm_readvariableop_2_resource*
_output_shapes
:S*
dtype0º
%batch_normalization_665/batchnorm/subSub:batch_normalization_665/batchnorm/ReadVariableOp_2:value:0+batch_normalization_665/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Sº
'batch_normalization_665/batchnorm/add_1AddV2+batch_normalization_665/batchnorm/mul_1:z:0)batch_normalization_665/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
leaky_re_lu_665/LeakyRelu	LeakyRelu+batch_normalization_665/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
alpha%>
dense_735/MatMul/ReadVariableOpReadVariableOp(dense_735_matmul_readvariableop_resource*
_output_shapes

:SS*
dtype0
dense_735/MatMulMatMul'leaky_re_lu_665/LeakyRelu:activations:0'dense_735/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 dense_735/BiasAdd/ReadVariableOpReadVariableOp)dense_735_biasadd_readvariableop_resource*
_output_shapes
:S*
dtype0
dense_735/BiasAddBiasAdddense_735/MatMul:product:0(dense_735/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS¦
0batch_normalization_666/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_666_batchnorm_readvariableop_resource*
_output_shapes
:S*
dtype0l
'batch_normalization_666/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_666/batchnorm/addAddV28batch_normalization_666/batchnorm/ReadVariableOp:value:00batch_normalization_666/batchnorm/add/y:output:0*
T0*
_output_shapes
:S
'batch_normalization_666/batchnorm/RsqrtRsqrt)batch_normalization_666/batchnorm/add:z:0*
T0*
_output_shapes
:S®
4batch_normalization_666/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_666_batchnorm_mul_readvariableop_resource*
_output_shapes
:S*
dtype0¼
%batch_normalization_666/batchnorm/mulMul+batch_normalization_666/batchnorm/Rsqrt:y:0<batch_normalization_666/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:S§
'batch_normalization_666/batchnorm/mul_1Muldense_735/BiasAdd:output:0)batch_normalization_666/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSª
2batch_normalization_666/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_666_batchnorm_readvariableop_1_resource*
_output_shapes
:S*
dtype0º
'batch_normalization_666/batchnorm/mul_2Mul:batch_normalization_666/batchnorm/ReadVariableOp_1:value:0)batch_normalization_666/batchnorm/mul:z:0*
T0*
_output_shapes
:Sª
2batch_normalization_666/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_666_batchnorm_readvariableop_2_resource*
_output_shapes
:S*
dtype0º
%batch_normalization_666/batchnorm/subSub:batch_normalization_666/batchnorm/ReadVariableOp_2:value:0+batch_normalization_666/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Sº
'batch_normalization_666/batchnorm/add_1AddV2+batch_normalization_666/batchnorm/mul_1:z:0)batch_normalization_666/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
leaky_re_lu_666/LeakyRelu	LeakyRelu+batch_normalization_666/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
alpha%>
dense_736/MatMul/ReadVariableOpReadVariableOp(dense_736_matmul_readvariableop_resource*
_output_shapes

:Sa*
dtype0
dense_736/MatMulMatMul'leaky_re_lu_666/LeakyRelu:activations:0'dense_736/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 dense_736/BiasAdd/ReadVariableOpReadVariableOp)dense_736_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
dense_736/BiasAddBiasAdddense_736/MatMul:product:0(dense_736/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¦
0batch_normalization_667/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_667_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0l
'batch_normalization_667/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_667/batchnorm/addAddV28batch_normalization_667/batchnorm/ReadVariableOp:value:00batch_normalization_667/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
'batch_normalization_667/batchnorm/RsqrtRsqrt)batch_normalization_667/batchnorm/add:z:0*
T0*
_output_shapes
:a®
4batch_normalization_667/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_667_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0¼
%batch_normalization_667/batchnorm/mulMul+batch_normalization_667/batchnorm/Rsqrt:y:0<batch_normalization_667/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:a§
'batch_normalization_667/batchnorm/mul_1Muldense_736/BiasAdd:output:0)batch_normalization_667/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaª
2batch_normalization_667/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_667_batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0º
'batch_normalization_667/batchnorm/mul_2Mul:batch_normalization_667/batchnorm/ReadVariableOp_1:value:0)batch_normalization_667/batchnorm/mul:z:0*
T0*
_output_shapes
:aª
2batch_normalization_667/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_667_batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0º
%batch_normalization_667/batchnorm/subSub:batch_normalization_667/batchnorm/ReadVariableOp_2:value:0+batch_normalization_667/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aº
'batch_normalization_667/batchnorm/add_1AddV2+batch_normalization_667/batchnorm/mul_1:z:0)batch_normalization_667/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
leaky_re_lu_667/LeakyRelu	LeakyRelu+batch_normalization_667/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>
dense_737/MatMul/ReadVariableOpReadVariableOp(dense_737_matmul_readvariableop_resource*
_output_shapes

:aa*
dtype0
dense_737/MatMulMatMul'leaky_re_lu_667/LeakyRelu:activations:0'dense_737/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
 dense_737/BiasAdd/ReadVariableOpReadVariableOp)dense_737_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
dense_737/BiasAddBiasAdddense_737/MatMul:product:0(dense_737/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa¦
0batch_normalization_668/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_668_batchnorm_readvariableop_resource*
_output_shapes
:a*
dtype0l
'batch_normalization_668/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_668/batchnorm/addAddV28batch_normalization_668/batchnorm/ReadVariableOp:value:00batch_normalization_668/batchnorm/add/y:output:0*
T0*
_output_shapes
:a
'batch_normalization_668/batchnorm/RsqrtRsqrt)batch_normalization_668/batchnorm/add:z:0*
T0*
_output_shapes
:a®
4batch_normalization_668/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_668_batchnorm_mul_readvariableop_resource*
_output_shapes
:a*
dtype0¼
%batch_normalization_668/batchnorm/mulMul+batch_normalization_668/batchnorm/Rsqrt:y:0<batch_normalization_668/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:a§
'batch_normalization_668/batchnorm/mul_1Muldense_737/BiasAdd:output:0)batch_normalization_668/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿaª
2batch_normalization_668/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_668_batchnorm_readvariableop_1_resource*
_output_shapes
:a*
dtype0º
'batch_normalization_668/batchnorm/mul_2Mul:batch_normalization_668/batchnorm/ReadVariableOp_1:value:0)batch_normalization_668/batchnorm/mul:z:0*
T0*
_output_shapes
:aª
2batch_normalization_668/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_668_batchnorm_readvariableop_2_resource*
_output_shapes
:a*
dtype0º
%batch_normalization_668/batchnorm/subSub:batch_normalization_668/batchnorm/ReadVariableOp_2:value:0+batch_normalization_668/batchnorm/mul_2:z:0*
T0*
_output_shapes
:aº
'batch_normalization_668/batchnorm/add_1AddV2+batch_normalization_668/batchnorm/mul_1:z:0)batch_normalization_668/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
leaky_re_lu_668/LeakyRelu	LeakyRelu+batch_normalization_668/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa*
alpha%>
dense_738/MatMul/ReadVariableOpReadVariableOp(dense_738_matmul_readvariableop_resource*
_output_shapes

:aA*
dtype0
dense_738/MatMulMatMul'leaky_re_lu_668/LeakyRelu:activations:0'dense_738/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_738/BiasAdd/ReadVariableOpReadVariableOp)dense_738_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_738/BiasAddBiasAdddense_738/MatMul:product:0(dense_738/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¦
0batch_normalization_669/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_669_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0l
'batch_normalization_669/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_669/batchnorm/addAddV28batch_normalization_669/batchnorm/ReadVariableOp:value:00batch_normalization_669/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
'batch_normalization_669/batchnorm/RsqrtRsqrt)batch_normalization_669/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_669/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_669_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_669/batchnorm/mulMul+batch_normalization_669/batchnorm/Rsqrt:y:0<batch_normalization_669/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_669/batchnorm/mul_1Muldense_738/BiasAdd:output:0)batch_normalization_669/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAª
2batch_normalization_669/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_669_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0º
'batch_normalization_669/batchnorm/mul_2Mul:batch_normalization_669/batchnorm/ReadVariableOp_1:value:0)batch_normalization_669/batchnorm/mul:z:0*
T0*
_output_shapes
:Aª
2batch_normalization_669/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_669_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0º
%batch_normalization_669/batchnorm/subSub:batch_normalization_669/batchnorm/ReadVariableOp_2:value:0+batch_normalization_669/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_669/batchnorm/add_1AddV2+batch_normalization_669/batchnorm/mul_1:z:0)batch_normalization_669/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_669/LeakyRelu	LeakyRelu+batch_normalization_669/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_739/MatMul/ReadVariableOpReadVariableOp(dense_739_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0
dense_739/MatMulMatMul'leaky_re_lu_669/LeakyRelu:activations:0'dense_739/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_739/BiasAdd/ReadVariableOpReadVariableOp)dense_739_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_739/BiasAddBiasAdddense_739/MatMul:product:0(dense_739/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¦
0batch_normalization_670/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_670_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0l
'batch_normalization_670/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_670/batchnorm/addAddV28batch_normalization_670/batchnorm/ReadVariableOp:value:00batch_normalization_670/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
'batch_normalization_670/batchnorm/RsqrtRsqrt)batch_normalization_670/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_670/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_670_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_670/batchnorm/mulMul+batch_normalization_670/batchnorm/Rsqrt:y:0<batch_normalization_670/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_670/batchnorm/mul_1Muldense_739/BiasAdd:output:0)batch_normalization_670/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAª
2batch_normalization_670/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_670_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0º
'batch_normalization_670/batchnorm/mul_2Mul:batch_normalization_670/batchnorm/ReadVariableOp_1:value:0)batch_normalization_670/batchnorm/mul:z:0*
T0*
_output_shapes
:Aª
2batch_normalization_670/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_670_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0º
%batch_normalization_670/batchnorm/subSub:batch_normalization_670/batchnorm/ReadVariableOp_2:value:0+batch_normalization_670/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_670/batchnorm/add_1AddV2+batch_normalization_670/batchnorm/mul_1:z:0)batch_normalization_670/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_670/LeakyRelu	LeakyRelu+batch_normalization_670/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_740/MatMul/ReadVariableOpReadVariableOp(dense_740_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0
dense_740/MatMulMatMul'leaky_re_lu_670/LeakyRelu:activations:0'dense_740/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_740/BiasAdd/ReadVariableOpReadVariableOp)dense_740_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_740/BiasAddBiasAdddense_740/MatMul:product:0(dense_740/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¦
0batch_normalization_671/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_671_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0l
'batch_normalization_671/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_671/batchnorm/addAddV28batch_normalization_671/batchnorm/ReadVariableOp:value:00batch_normalization_671/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
'batch_normalization_671/batchnorm/RsqrtRsqrt)batch_normalization_671/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_671/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_671_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_671/batchnorm/mulMul+batch_normalization_671/batchnorm/Rsqrt:y:0<batch_normalization_671/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_671/batchnorm/mul_1Muldense_740/BiasAdd:output:0)batch_normalization_671/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAª
2batch_normalization_671/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_671_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0º
'batch_normalization_671/batchnorm/mul_2Mul:batch_normalization_671/batchnorm/ReadVariableOp_1:value:0)batch_normalization_671/batchnorm/mul:z:0*
T0*
_output_shapes
:Aª
2batch_normalization_671/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_671_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0º
%batch_normalization_671/batchnorm/subSub:batch_normalization_671/batchnorm/ReadVariableOp_2:value:0+batch_normalization_671/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_671/batchnorm/add_1AddV2+batch_normalization_671/batchnorm/mul_1:z:0)batch_normalization_671/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_671/LeakyRelu	LeakyRelu+batch_normalization_671/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_741/MatMul/ReadVariableOpReadVariableOp(dense_741_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0
dense_741/MatMulMatMul'leaky_re_lu_671/LeakyRelu:activations:0'dense_741/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_741/BiasAdd/ReadVariableOpReadVariableOp)dense_741_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_741/BiasAddBiasAdddense_741/MatMul:product:0(dense_741/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¦
0batch_normalization_672/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_672_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0l
'batch_normalization_672/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_672/batchnorm/addAddV28batch_normalization_672/batchnorm/ReadVariableOp:value:00batch_normalization_672/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
'batch_normalization_672/batchnorm/RsqrtRsqrt)batch_normalization_672/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_672/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_672_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_672/batchnorm/mulMul+batch_normalization_672/batchnorm/Rsqrt:y:0<batch_normalization_672/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_672/batchnorm/mul_1Muldense_741/BiasAdd:output:0)batch_normalization_672/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAª
2batch_normalization_672/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_672_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0º
'batch_normalization_672/batchnorm/mul_2Mul:batch_normalization_672/batchnorm/ReadVariableOp_1:value:0)batch_normalization_672/batchnorm/mul:z:0*
T0*
_output_shapes
:Aª
2batch_normalization_672/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_672_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0º
%batch_normalization_672/batchnorm/subSub:batch_normalization_672/batchnorm/ReadVariableOp_2:value:0+batch_normalization_672/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_672/batchnorm/add_1AddV2+batch_normalization_672/batchnorm/mul_1:z:0)batch_normalization_672/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_672/LeakyRelu	LeakyRelu+batch_normalization_672/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_742/MatMul/ReadVariableOpReadVariableOp(dense_742_matmul_readvariableop_resource*
_output_shapes

:AA*
dtype0
dense_742/MatMulMatMul'leaky_re_lu_672/LeakyRelu:activations:0'dense_742/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
 dense_742/BiasAdd/ReadVariableOpReadVariableOp)dense_742_biasadd_readvariableop_resource*
_output_shapes
:A*
dtype0
dense_742/BiasAddBiasAdddense_742/MatMul:product:0(dense_742/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA¦
0batch_normalization_673/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_673_batchnorm_readvariableop_resource*
_output_shapes
:A*
dtype0l
'batch_normalization_673/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_673/batchnorm/addAddV28batch_normalization_673/batchnorm/ReadVariableOp:value:00batch_normalization_673/batchnorm/add/y:output:0*
T0*
_output_shapes
:A
'batch_normalization_673/batchnorm/RsqrtRsqrt)batch_normalization_673/batchnorm/add:z:0*
T0*
_output_shapes
:A®
4batch_normalization_673/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_673_batchnorm_mul_readvariableop_resource*
_output_shapes
:A*
dtype0¼
%batch_normalization_673/batchnorm/mulMul+batch_normalization_673/batchnorm/Rsqrt:y:0<batch_normalization_673/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:A§
'batch_normalization_673/batchnorm/mul_1Muldense_742/BiasAdd:output:0)batch_normalization_673/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿAª
2batch_normalization_673/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_673_batchnorm_readvariableop_1_resource*
_output_shapes
:A*
dtype0º
'batch_normalization_673/batchnorm/mul_2Mul:batch_normalization_673/batchnorm/ReadVariableOp_1:value:0)batch_normalization_673/batchnorm/mul:z:0*
T0*
_output_shapes
:Aª
2batch_normalization_673/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_673_batchnorm_readvariableop_2_resource*
_output_shapes
:A*
dtype0º
%batch_normalization_673/batchnorm/subSub:batch_normalization_673/batchnorm/ReadVariableOp_2:value:0+batch_normalization_673/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Aº
'batch_normalization_673/batchnorm/add_1AddV2+batch_normalization_673/batchnorm/mul_1:z:0)batch_normalization_673/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA
leaky_re_lu_673/LeakyRelu	LeakyRelu+batch_normalization_673/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿA*
alpha%>
dense_743/MatMul/ReadVariableOpReadVariableOp(dense_743_matmul_readvariableop_resource*
_output_shapes

:A*
dtype0
dense_743/MatMulMatMul'leaky_re_lu_673/LeakyRelu:activations:0'dense_743/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_743/BiasAdd/ReadVariableOpReadVariableOp)dense_743_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_743/BiasAddBiasAdddense_743/MatMul:product:0(dense_743/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_743/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
NoOpNoOp1^batch_normalization_665/batchnorm/ReadVariableOp3^batch_normalization_665/batchnorm/ReadVariableOp_13^batch_normalization_665/batchnorm/ReadVariableOp_25^batch_normalization_665/batchnorm/mul/ReadVariableOp1^batch_normalization_666/batchnorm/ReadVariableOp3^batch_normalization_666/batchnorm/ReadVariableOp_13^batch_normalization_666/batchnorm/ReadVariableOp_25^batch_normalization_666/batchnorm/mul/ReadVariableOp1^batch_normalization_667/batchnorm/ReadVariableOp3^batch_normalization_667/batchnorm/ReadVariableOp_13^batch_normalization_667/batchnorm/ReadVariableOp_25^batch_normalization_667/batchnorm/mul/ReadVariableOp1^batch_normalization_668/batchnorm/ReadVariableOp3^batch_normalization_668/batchnorm/ReadVariableOp_13^batch_normalization_668/batchnorm/ReadVariableOp_25^batch_normalization_668/batchnorm/mul/ReadVariableOp1^batch_normalization_669/batchnorm/ReadVariableOp3^batch_normalization_669/batchnorm/ReadVariableOp_13^batch_normalization_669/batchnorm/ReadVariableOp_25^batch_normalization_669/batchnorm/mul/ReadVariableOp1^batch_normalization_670/batchnorm/ReadVariableOp3^batch_normalization_670/batchnorm/ReadVariableOp_13^batch_normalization_670/batchnorm/ReadVariableOp_25^batch_normalization_670/batchnorm/mul/ReadVariableOp1^batch_normalization_671/batchnorm/ReadVariableOp3^batch_normalization_671/batchnorm/ReadVariableOp_13^batch_normalization_671/batchnorm/ReadVariableOp_25^batch_normalization_671/batchnorm/mul/ReadVariableOp1^batch_normalization_672/batchnorm/ReadVariableOp3^batch_normalization_672/batchnorm/ReadVariableOp_13^batch_normalization_672/batchnorm/ReadVariableOp_25^batch_normalization_672/batchnorm/mul/ReadVariableOp1^batch_normalization_673/batchnorm/ReadVariableOp3^batch_normalization_673/batchnorm/ReadVariableOp_13^batch_normalization_673/batchnorm/ReadVariableOp_25^batch_normalization_673/batchnorm/mul/ReadVariableOp!^dense_734/BiasAdd/ReadVariableOp ^dense_734/MatMul/ReadVariableOp!^dense_735/BiasAdd/ReadVariableOp ^dense_735/MatMul/ReadVariableOp!^dense_736/BiasAdd/ReadVariableOp ^dense_736/MatMul/ReadVariableOp!^dense_737/BiasAdd/ReadVariableOp ^dense_737/MatMul/ReadVariableOp!^dense_738/BiasAdd/ReadVariableOp ^dense_738/MatMul/ReadVariableOp!^dense_739/BiasAdd/ReadVariableOp ^dense_739/MatMul/ReadVariableOp!^dense_740/BiasAdd/ReadVariableOp ^dense_740/MatMul/ReadVariableOp!^dense_741/BiasAdd/ReadVariableOp ^dense_741/MatMul/ReadVariableOp!^dense_742/BiasAdd/ReadVariableOp ^dense_742/MatMul/ReadVariableOp!^dense_743/BiasAdd/ReadVariableOp ^dense_743/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0batch_normalization_665/batchnorm/ReadVariableOp0batch_normalization_665/batchnorm/ReadVariableOp2h
2batch_normalization_665/batchnorm/ReadVariableOp_12batch_normalization_665/batchnorm/ReadVariableOp_12h
2batch_normalization_665/batchnorm/ReadVariableOp_22batch_normalization_665/batchnorm/ReadVariableOp_22l
4batch_normalization_665/batchnorm/mul/ReadVariableOp4batch_normalization_665/batchnorm/mul/ReadVariableOp2d
0batch_normalization_666/batchnorm/ReadVariableOp0batch_normalization_666/batchnorm/ReadVariableOp2h
2batch_normalization_666/batchnorm/ReadVariableOp_12batch_normalization_666/batchnorm/ReadVariableOp_12h
2batch_normalization_666/batchnorm/ReadVariableOp_22batch_normalization_666/batchnorm/ReadVariableOp_22l
4batch_normalization_666/batchnorm/mul/ReadVariableOp4batch_normalization_666/batchnorm/mul/ReadVariableOp2d
0batch_normalization_667/batchnorm/ReadVariableOp0batch_normalization_667/batchnorm/ReadVariableOp2h
2batch_normalization_667/batchnorm/ReadVariableOp_12batch_normalization_667/batchnorm/ReadVariableOp_12h
2batch_normalization_667/batchnorm/ReadVariableOp_22batch_normalization_667/batchnorm/ReadVariableOp_22l
4batch_normalization_667/batchnorm/mul/ReadVariableOp4batch_normalization_667/batchnorm/mul/ReadVariableOp2d
0batch_normalization_668/batchnorm/ReadVariableOp0batch_normalization_668/batchnorm/ReadVariableOp2h
2batch_normalization_668/batchnorm/ReadVariableOp_12batch_normalization_668/batchnorm/ReadVariableOp_12h
2batch_normalization_668/batchnorm/ReadVariableOp_22batch_normalization_668/batchnorm/ReadVariableOp_22l
4batch_normalization_668/batchnorm/mul/ReadVariableOp4batch_normalization_668/batchnorm/mul/ReadVariableOp2d
0batch_normalization_669/batchnorm/ReadVariableOp0batch_normalization_669/batchnorm/ReadVariableOp2h
2batch_normalization_669/batchnorm/ReadVariableOp_12batch_normalization_669/batchnorm/ReadVariableOp_12h
2batch_normalization_669/batchnorm/ReadVariableOp_22batch_normalization_669/batchnorm/ReadVariableOp_22l
4batch_normalization_669/batchnorm/mul/ReadVariableOp4batch_normalization_669/batchnorm/mul/ReadVariableOp2d
0batch_normalization_670/batchnorm/ReadVariableOp0batch_normalization_670/batchnorm/ReadVariableOp2h
2batch_normalization_670/batchnorm/ReadVariableOp_12batch_normalization_670/batchnorm/ReadVariableOp_12h
2batch_normalization_670/batchnorm/ReadVariableOp_22batch_normalization_670/batchnorm/ReadVariableOp_22l
4batch_normalization_670/batchnorm/mul/ReadVariableOp4batch_normalization_670/batchnorm/mul/ReadVariableOp2d
0batch_normalization_671/batchnorm/ReadVariableOp0batch_normalization_671/batchnorm/ReadVariableOp2h
2batch_normalization_671/batchnorm/ReadVariableOp_12batch_normalization_671/batchnorm/ReadVariableOp_12h
2batch_normalization_671/batchnorm/ReadVariableOp_22batch_normalization_671/batchnorm/ReadVariableOp_22l
4batch_normalization_671/batchnorm/mul/ReadVariableOp4batch_normalization_671/batchnorm/mul/ReadVariableOp2d
0batch_normalization_672/batchnorm/ReadVariableOp0batch_normalization_672/batchnorm/ReadVariableOp2h
2batch_normalization_672/batchnorm/ReadVariableOp_12batch_normalization_672/batchnorm/ReadVariableOp_12h
2batch_normalization_672/batchnorm/ReadVariableOp_22batch_normalization_672/batchnorm/ReadVariableOp_22l
4batch_normalization_672/batchnorm/mul/ReadVariableOp4batch_normalization_672/batchnorm/mul/ReadVariableOp2d
0batch_normalization_673/batchnorm/ReadVariableOp0batch_normalization_673/batchnorm/ReadVariableOp2h
2batch_normalization_673/batchnorm/ReadVariableOp_12batch_normalization_673/batchnorm/ReadVariableOp_12h
2batch_normalization_673/batchnorm/ReadVariableOp_22batch_normalization_673/batchnorm/ReadVariableOp_22l
4batch_normalization_673/batchnorm/mul/ReadVariableOp4batch_normalization_673/batchnorm/mul/ReadVariableOp2D
 dense_734/BiasAdd/ReadVariableOp dense_734/BiasAdd/ReadVariableOp2B
dense_734/MatMul/ReadVariableOpdense_734/MatMul/ReadVariableOp2D
 dense_735/BiasAdd/ReadVariableOp dense_735/BiasAdd/ReadVariableOp2B
dense_735/MatMul/ReadVariableOpdense_735/MatMul/ReadVariableOp2D
 dense_736/BiasAdd/ReadVariableOp dense_736/BiasAdd/ReadVariableOp2B
dense_736/MatMul/ReadVariableOpdense_736/MatMul/ReadVariableOp2D
 dense_737/BiasAdd/ReadVariableOp dense_737/BiasAdd/ReadVariableOp2B
dense_737/MatMul/ReadVariableOpdense_737/MatMul/ReadVariableOp2D
 dense_738/BiasAdd/ReadVariableOp dense_738/BiasAdd/ReadVariableOp2B
dense_738/MatMul/ReadVariableOpdense_738/MatMul/ReadVariableOp2D
 dense_739/BiasAdd/ReadVariableOp dense_739/BiasAdd/ReadVariableOp2B
dense_739/MatMul/ReadVariableOpdense_739/MatMul/ReadVariableOp2D
 dense_740/BiasAdd/ReadVariableOp dense_740/BiasAdd/ReadVariableOp2B
dense_740/MatMul/ReadVariableOpdense_740/MatMul/ReadVariableOp2D
 dense_741/BiasAdd/ReadVariableOp dense_741/BiasAdd/ReadVariableOp2B
dense_741/MatMul/ReadVariableOpdense_741/MatMul/ReadVariableOp2D
 dense_742/BiasAdd/ReadVariableOp dense_742/BiasAdd/ReadVariableOp2B
dense_742/MatMul/ReadVariableOpdense_742/MatMul/ReadVariableOp2D
 dense_743/BiasAdd/ReadVariableOp dense_743/BiasAdd/ReadVariableOp2B
dense_743/MatMul/ReadVariableOpdense_743/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
«
L
0__inference_leaky_re_lu_665_layer_call_fn_736233

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
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_665_layer_call_and_return_conditional_losses_733775`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿS:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
Ã
û
I__inference_sequential_69_layer_call_and_return_conditional_losses_735139
normalization_69_input
normalization_69_sub_y
normalization_69_sqrt_x"
dense_734_734998:S
dense_734_735000:S,
batch_normalization_665_735003:S,
batch_normalization_665_735005:S,
batch_normalization_665_735007:S,
batch_normalization_665_735009:S"
dense_735_735013:SS
dense_735_735015:S,
batch_normalization_666_735018:S,
batch_normalization_666_735020:S,
batch_normalization_666_735022:S,
batch_normalization_666_735024:S"
dense_736_735028:Sa
dense_736_735030:a,
batch_normalization_667_735033:a,
batch_normalization_667_735035:a,
batch_normalization_667_735037:a,
batch_normalization_667_735039:a"
dense_737_735043:aa
dense_737_735045:a,
batch_normalization_668_735048:a,
batch_normalization_668_735050:a,
batch_normalization_668_735052:a,
batch_normalization_668_735054:a"
dense_738_735058:aA
dense_738_735060:A,
batch_normalization_669_735063:A,
batch_normalization_669_735065:A,
batch_normalization_669_735067:A,
batch_normalization_669_735069:A"
dense_739_735073:AA
dense_739_735075:A,
batch_normalization_670_735078:A,
batch_normalization_670_735080:A,
batch_normalization_670_735082:A,
batch_normalization_670_735084:A"
dense_740_735088:AA
dense_740_735090:A,
batch_normalization_671_735093:A,
batch_normalization_671_735095:A,
batch_normalization_671_735097:A,
batch_normalization_671_735099:A"
dense_741_735103:AA
dense_741_735105:A,
batch_normalization_672_735108:A,
batch_normalization_672_735110:A,
batch_normalization_672_735112:A,
batch_normalization_672_735114:A"
dense_742_735118:AA
dense_742_735120:A,
batch_normalization_673_735123:A,
batch_normalization_673_735125:A,
batch_normalization_673_735127:A,
batch_normalization_673_735129:A"
dense_743_735133:A
dense_743_735135:
identity¢/batch_normalization_665/StatefulPartitionedCall¢/batch_normalization_666/StatefulPartitionedCall¢/batch_normalization_667/StatefulPartitionedCall¢/batch_normalization_668/StatefulPartitionedCall¢/batch_normalization_669/StatefulPartitionedCall¢/batch_normalization_670/StatefulPartitionedCall¢/batch_normalization_671/StatefulPartitionedCall¢/batch_normalization_672/StatefulPartitionedCall¢/batch_normalization_673/StatefulPartitionedCall¢!dense_734/StatefulPartitionedCall¢!dense_735/StatefulPartitionedCall¢!dense_736/StatefulPartitionedCall¢!dense_737/StatefulPartitionedCall¢!dense_738/StatefulPartitionedCall¢!dense_739/StatefulPartitionedCall¢!dense_740/StatefulPartitionedCall¢!dense_741/StatefulPartitionedCall¢!dense_742/StatefulPartitionedCall¢!dense_743/StatefulPartitionedCall}
normalization_69/subSubnormalization_69_inputnormalization_69_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_69/SqrtSqrtnormalization_69_sqrt_x*
T0*
_output_shapes

:_
normalization_69/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_69/MaximumMaximumnormalization_69/Sqrt:y:0#normalization_69/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_69/truedivRealDivnormalization_69/sub:z:0normalization_69/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_734/StatefulPartitionedCallStatefulPartitionedCallnormalization_69/truediv:z:0dense_734_734998dense_734_735000*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_734_layer_call_and_return_conditional_losses_733755
/batch_normalization_665/StatefulPartitionedCallStatefulPartitionedCall*dense_734/StatefulPartitionedCall:output:0batch_normalization_665_735003batch_normalization_665_735005batch_normalization_665_735007batch_normalization_665_735009*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_665_layer_call_and_return_conditional_losses_733064ø
leaky_re_lu_665/PartitionedCallPartitionedCall8batch_normalization_665/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_665_layer_call_and_return_conditional_losses_733775
!dense_735/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_665/PartitionedCall:output:0dense_735_735013dense_735_735015*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_735_layer_call_and_return_conditional_losses_733787
/batch_normalization_666/StatefulPartitionedCallStatefulPartitionedCall*dense_735/StatefulPartitionedCall:output:0batch_normalization_666_735018batch_normalization_666_735020batch_normalization_666_735022batch_normalization_666_735024*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_666_layer_call_and_return_conditional_losses_733146ø
leaky_re_lu_666/PartitionedCallPartitionedCall8batch_normalization_666/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_666_layer_call_and_return_conditional_losses_733807
!dense_736/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_666/PartitionedCall:output:0dense_736_735028dense_736_735030*
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
E__inference_dense_736_layer_call_and_return_conditional_losses_733819
/batch_normalization_667/StatefulPartitionedCallStatefulPartitionedCall*dense_736/StatefulPartitionedCall:output:0batch_normalization_667_735033batch_normalization_667_735035batch_normalization_667_735037batch_normalization_667_735039*
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
S__inference_batch_normalization_667_layer_call_and_return_conditional_losses_733228ø
leaky_re_lu_667/PartitionedCallPartitionedCall8batch_normalization_667/StatefulPartitionedCall:output:0*
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
K__inference_leaky_re_lu_667_layer_call_and_return_conditional_losses_733839
!dense_737/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_667/PartitionedCall:output:0dense_737_735043dense_737_735045*
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
E__inference_dense_737_layer_call_and_return_conditional_losses_733851
/batch_normalization_668/StatefulPartitionedCallStatefulPartitionedCall*dense_737/StatefulPartitionedCall:output:0batch_normalization_668_735048batch_normalization_668_735050batch_normalization_668_735052batch_normalization_668_735054*
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
S__inference_batch_normalization_668_layer_call_and_return_conditional_losses_733310ø
leaky_re_lu_668/PartitionedCallPartitionedCall8batch_normalization_668/StatefulPartitionedCall:output:0*
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
K__inference_leaky_re_lu_668_layer_call_and_return_conditional_losses_733871
!dense_738/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_668/PartitionedCall:output:0dense_738_735058dense_738_735060*
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
GPU 2J 8 *N
fIRG
E__inference_dense_738_layer_call_and_return_conditional_losses_733883
/batch_normalization_669/StatefulPartitionedCallStatefulPartitionedCall*dense_738/StatefulPartitionedCall:output:0batch_normalization_669_735063batch_normalization_669_735065batch_normalization_669_735067batch_normalization_669_735069*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_669_layer_call_and_return_conditional_losses_733392ø
leaky_re_lu_669/PartitionedCallPartitionedCall8batch_normalization_669/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_669_layer_call_and_return_conditional_losses_733903
!dense_739/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_669/PartitionedCall:output:0dense_739_735073dense_739_735075*
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
GPU 2J 8 *N
fIRG
E__inference_dense_739_layer_call_and_return_conditional_losses_733915
/batch_normalization_670/StatefulPartitionedCallStatefulPartitionedCall*dense_739/StatefulPartitionedCall:output:0batch_normalization_670_735078batch_normalization_670_735080batch_normalization_670_735082batch_normalization_670_735084*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_670_layer_call_and_return_conditional_losses_733474ø
leaky_re_lu_670/PartitionedCallPartitionedCall8batch_normalization_670/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_670_layer_call_and_return_conditional_losses_733935
!dense_740/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_670/PartitionedCall:output:0dense_740_735088dense_740_735090*
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
GPU 2J 8 *N
fIRG
E__inference_dense_740_layer_call_and_return_conditional_losses_733947
/batch_normalization_671/StatefulPartitionedCallStatefulPartitionedCall*dense_740/StatefulPartitionedCall:output:0batch_normalization_671_735093batch_normalization_671_735095batch_normalization_671_735097batch_normalization_671_735099*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_671_layer_call_and_return_conditional_losses_733556ø
leaky_re_lu_671/PartitionedCallPartitionedCall8batch_normalization_671/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_671_layer_call_and_return_conditional_losses_733967
!dense_741/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_671/PartitionedCall:output:0dense_741_735103dense_741_735105*
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
GPU 2J 8 *N
fIRG
E__inference_dense_741_layer_call_and_return_conditional_losses_733979
/batch_normalization_672/StatefulPartitionedCallStatefulPartitionedCall*dense_741/StatefulPartitionedCall:output:0batch_normalization_672_735108batch_normalization_672_735110batch_normalization_672_735112batch_normalization_672_735114*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_672_layer_call_and_return_conditional_losses_733638ø
leaky_re_lu_672/PartitionedCallPartitionedCall8batch_normalization_672/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_672_layer_call_and_return_conditional_losses_733999
!dense_742/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_672/PartitionedCall:output:0dense_742_735118dense_742_735120*
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
GPU 2J 8 *N
fIRG
E__inference_dense_742_layer_call_and_return_conditional_losses_734011
/batch_normalization_673/StatefulPartitionedCallStatefulPartitionedCall*dense_742/StatefulPartitionedCall:output:0batch_normalization_673_735123batch_normalization_673_735125batch_normalization_673_735127batch_normalization_673_735129*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_673_layer_call_and_return_conditional_losses_733720ø
leaky_re_lu_673/PartitionedCallPartitionedCall8batch_normalization_673/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_673_layer_call_and_return_conditional_losses_734031
!dense_743/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_673/PartitionedCall:output:0dense_743_735133dense_743_735135*
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
E__inference_dense_743_layer_call_and_return_conditional_losses_734043y
IdentityIdentity*dense_743/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp0^batch_normalization_665/StatefulPartitionedCall0^batch_normalization_666/StatefulPartitionedCall0^batch_normalization_667/StatefulPartitionedCall0^batch_normalization_668/StatefulPartitionedCall0^batch_normalization_669/StatefulPartitionedCall0^batch_normalization_670/StatefulPartitionedCall0^batch_normalization_671/StatefulPartitionedCall0^batch_normalization_672/StatefulPartitionedCall0^batch_normalization_673/StatefulPartitionedCall"^dense_734/StatefulPartitionedCall"^dense_735/StatefulPartitionedCall"^dense_736/StatefulPartitionedCall"^dense_737/StatefulPartitionedCall"^dense_738/StatefulPartitionedCall"^dense_739/StatefulPartitionedCall"^dense_740/StatefulPartitionedCall"^dense_741/StatefulPartitionedCall"^dense_742/StatefulPartitionedCall"^dense_743/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¬
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_665/StatefulPartitionedCall/batch_normalization_665/StatefulPartitionedCall2b
/batch_normalization_666/StatefulPartitionedCall/batch_normalization_666/StatefulPartitionedCall2b
/batch_normalization_667/StatefulPartitionedCall/batch_normalization_667/StatefulPartitionedCall2b
/batch_normalization_668/StatefulPartitionedCall/batch_normalization_668/StatefulPartitionedCall2b
/batch_normalization_669/StatefulPartitionedCall/batch_normalization_669/StatefulPartitionedCall2b
/batch_normalization_670/StatefulPartitionedCall/batch_normalization_670/StatefulPartitionedCall2b
/batch_normalization_671/StatefulPartitionedCall/batch_normalization_671/StatefulPartitionedCall2b
/batch_normalization_672/StatefulPartitionedCall/batch_normalization_672/StatefulPartitionedCall2b
/batch_normalization_673/StatefulPartitionedCall/batch_normalization_673/StatefulPartitionedCall2F
!dense_734/StatefulPartitionedCall!dense_734/StatefulPartitionedCall2F
!dense_735/StatefulPartitionedCall!dense_735/StatefulPartitionedCall2F
!dense_736/StatefulPartitionedCall!dense_736/StatefulPartitionedCall2F
!dense_737/StatefulPartitionedCall!dense_737/StatefulPartitionedCall2F
!dense_738/StatefulPartitionedCall!dense_738/StatefulPartitionedCall2F
!dense_739/StatefulPartitionedCall!dense_739/StatefulPartitionedCall2F
!dense_740/StatefulPartitionedCall!dense_740/StatefulPartitionedCall2F
!dense_741/StatefulPartitionedCall!dense_741/StatefulPartitionedCall2F
!dense_742/StatefulPartitionedCall!dense_742/StatefulPartitionedCall2F
!dense_743/StatefulPartitionedCall!dense_743/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_69_input:$ 

_output_shapes

::$ 

_output_shapes

:
¬
Ó
8__inference_batch_normalization_671_layer_call_fn_736815

inputs
unknown:A
	unknown_0:A
	unknown_1:A
	unknown_2:A
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_671_layer_call_and_return_conditional_losses_733509o
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
«
L
0__inference_leaky_re_lu_666_layer_call_fn_736342

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
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_666_layer_call_and_return_conditional_losses_733807`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿS:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_667_layer_call_fn_736392

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
S__inference_batch_normalization_667_layer_call_and_return_conditional_losses_733228o
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
S__inference_batch_normalization_670_layer_call_and_return_conditional_losses_733427

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
Ð
²
S__inference_batch_normalization_668_layer_call_and_return_conditional_losses_733263

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
K__inference_leaky_re_lu_672_layer_call_and_return_conditional_losses_733999

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
å
g
K__inference_leaky_re_lu_665_layer_call_and_return_conditional_losses_736238

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿS:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
È	
ö
E__inference_dense_739_layer_call_and_return_conditional_losses_736693

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
È	
ö
E__inference_dense_734_layer_call_and_return_conditional_losses_733755

inputs0
matmul_readvariableop_resource:S-
biasadd_readvariableop_resource:S
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:S*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:S*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿSw
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
normalization_69_input?
(serving_default_normalization_69_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_7430
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ø
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
.__inference_sequential_69_layer_call_fn_734169
.__inference_sequential_69_layer_call_fn_735264
.__inference_sequential_69_layer_call_fn_735385
.__inference_sequential_69_layer_call_fn_734837À
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
I__inference_sequential_69_layer_call_and_return_conditional_losses_735609
I__inference_sequential_69_layer_call_and_return_conditional_losses_735959
I__inference_sequential_69_layer_call_and_return_conditional_losses_734988
I__inference_sequential_69_layer_call_and_return_conditional_losses_735139À
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
!__inference__wrapped_model_732993normalization_69_input"
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
:2mean
:2variance
:	 2count
"
_generic_user_object
¿2¼
__inference_adapt_step_736129
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
": S2dense_734/kernel
:S2dense_734/bias
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
*__inference_dense_734_layer_call_fn_736138¢
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
E__inference_dense_734_layer_call_and_return_conditional_losses_736148¢
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
+:)S2batch_normalization_665/gamma
*:(S2batch_normalization_665/beta
3:1S (2#batch_normalization_665/moving_mean
7:5S (2'batch_normalization_665/moving_variance
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
8__inference_batch_normalization_665_layer_call_fn_736161
8__inference_batch_normalization_665_layer_call_fn_736174´
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
S__inference_batch_normalization_665_layer_call_and_return_conditional_losses_736194
S__inference_batch_normalization_665_layer_call_and_return_conditional_losses_736228´
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
0__inference_leaky_re_lu_665_layer_call_fn_736233¢
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
K__inference_leaky_re_lu_665_layer_call_and_return_conditional_losses_736238¢
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
": SS2dense_735/kernel
:S2dense_735/bias
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
*__inference_dense_735_layer_call_fn_736247¢
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
E__inference_dense_735_layer_call_and_return_conditional_losses_736257¢
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
+:)S2batch_normalization_666/gamma
*:(S2batch_normalization_666/beta
3:1S (2#batch_normalization_666/moving_mean
7:5S (2'batch_normalization_666/moving_variance
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
8__inference_batch_normalization_666_layer_call_fn_736270
8__inference_batch_normalization_666_layer_call_fn_736283´
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
S__inference_batch_normalization_666_layer_call_and_return_conditional_losses_736303
S__inference_batch_normalization_666_layer_call_and_return_conditional_losses_736337´
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
0__inference_leaky_re_lu_666_layer_call_fn_736342¢
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
K__inference_leaky_re_lu_666_layer_call_and_return_conditional_losses_736347¢
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
": Sa2dense_736/kernel
:a2dense_736/bias
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
*__inference_dense_736_layer_call_fn_736356¢
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
E__inference_dense_736_layer_call_and_return_conditional_losses_736366¢
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
+:)a2batch_normalization_667/gamma
*:(a2batch_normalization_667/beta
3:1a (2#batch_normalization_667/moving_mean
7:5a (2'batch_normalization_667/moving_variance
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
8__inference_batch_normalization_667_layer_call_fn_736379
8__inference_batch_normalization_667_layer_call_fn_736392´
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
S__inference_batch_normalization_667_layer_call_and_return_conditional_losses_736412
S__inference_batch_normalization_667_layer_call_and_return_conditional_losses_736446´
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
0__inference_leaky_re_lu_667_layer_call_fn_736451¢
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
K__inference_leaky_re_lu_667_layer_call_and_return_conditional_losses_736456¢
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
": aa2dense_737/kernel
:a2dense_737/bias
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
*__inference_dense_737_layer_call_fn_736465¢
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
E__inference_dense_737_layer_call_and_return_conditional_losses_736475¢
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
+:)a2batch_normalization_668/gamma
*:(a2batch_normalization_668/beta
3:1a (2#batch_normalization_668/moving_mean
7:5a (2'batch_normalization_668/moving_variance
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
8__inference_batch_normalization_668_layer_call_fn_736488
8__inference_batch_normalization_668_layer_call_fn_736501´
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
S__inference_batch_normalization_668_layer_call_and_return_conditional_losses_736521
S__inference_batch_normalization_668_layer_call_and_return_conditional_losses_736555´
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
0__inference_leaky_re_lu_668_layer_call_fn_736560¢
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
K__inference_leaky_re_lu_668_layer_call_and_return_conditional_losses_736565¢
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
": aA2dense_738/kernel
:A2dense_738/bias
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
*__inference_dense_738_layer_call_fn_736574¢
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
E__inference_dense_738_layer_call_and_return_conditional_losses_736584¢
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
+:)A2batch_normalization_669/gamma
*:(A2batch_normalization_669/beta
3:1A (2#batch_normalization_669/moving_mean
7:5A (2'batch_normalization_669/moving_variance
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
8__inference_batch_normalization_669_layer_call_fn_736597
8__inference_batch_normalization_669_layer_call_fn_736610´
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
S__inference_batch_normalization_669_layer_call_and_return_conditional_losses_736630
S__inference_batch_normalization_669_layer_call_and_return_conditional_losses_736664´
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
0__inference_leaky_re_lu_669_layer_call_fn_736669¢
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
K__inference_leaky_re_lu_669_layer_call_and_return_conditional_losses_736674¢
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
": AA2dense_739/kernel
:A2dense_739/bias
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
*__inference_dense_739_layer_call_fn_736683¢
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
E__inference_dense_739_layer_call_and_return_conditional_losses_736693¢
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
+:)A2batch_normalization_670/gamma
*:(A2batch_normalization_670/beta
3:1A (2#batch_normalization_670/moving_mean
7:5A (2'batch_normalization_670/moving_variance
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
8__inference_batch_normalization_670_layer_call_fn_736706
8__inference_batch_normalization_670_layer_call_fn_736719´
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
S__inference_batch_normalization_670_layer_call_and_return_conditional_losses_736739
S__inference_batch_normalization_670_layer_call_and_return_conditional_losses_736773´
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
0__inference_leaky_re_lu_670_layer_call_fn_736778¢
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
K__inference_leaky_re_lu_670_layer_call_and_return_conditional_losses_736783¢
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
": AA2dense_740/kernel
:A2dense_740/bias
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
*__inference_dense_740_layer_call_fn_736792¢
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
E__inference_dense_740_layer_call_and_return_conditional_losses_736802¢
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
+:)A2batch_normalization_671/gamma
*:(A2batch_normalization_671/beta
3:1A (2#batch_normalization_671/moving_mean
7:5A (2'batch_normalization_671/moving_variance
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
8__inference_batch_normalization_671_layer_call_fn_736815
8__inference_batch_normalization_671_layer_call_fn_736828´
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
S__inference_batch_normalization_671_layer_call_and_return_conditional_losses_736848
S__inference_batch_normalization_671_layer_call_and_return_conditional_losses_736882´
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
0__inference_leaky_re_lu_671_layer_call_fn_736887¢
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
K__inference_leaky_re_lu_671_layer_call_and_return_conditional_losses_736892¢
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
": AA2dense_741/kernel
:A2dense_741/bias
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
*__inference_dense_741_layer_call_fn_736901¢
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
E__inference_dense_741_layer_call_and_return_conditional_losses_736911¢
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
+:)A2batch_normalization_672/gamma
*:(A2batch_normalization_672/beta
3:1A (2#batch_normalization_672/moving_mean
7:5A (2'batch_normalization_672/moving_variance
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
8__inference_batch_normalization_672_layer_call_fn_736924
8__inference_batch_normalization_672_layer_call_fn_736937´
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
S__inference_batch_normalization_672_layer_call_and_return_conditional_losses_736957
S__inference_batch_normalization_672_layer_call_and_return_conditional_losses_736991´
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
0__inference_leaky_re_lu_672_layer_call_fn_736996¢
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
K__inference_leaky_re_lu_672_layer_call_and_return_conditional_losses_737001¢
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
": AA2dense_742/kernel
:A2dense_742/bias
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
*__inference_dense_742_layer_call_fn_737010¢
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
E__inference_dense_742_layer_call_and_return_conditional_losses_737020¢
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
+:)A2batch_normalization_673/gamma
*:(A2batch_normalization_673/beta
3:1A (2#batch_normalization_673/moving_mean
7:5A (2'batch_normalization_673/moving_variance
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
8__inference_batch_normalization_673_layer_call_fn_737033
8__inference_batch_normalization_673_layer_call_fn_737046´
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
S__inference_batch_normalization_673_layer_call_and_return_conditional_losses_737066
S__inference_batch_normalization_673_layer_call_and_return_conditional_losses_737100´
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
0__inference_leaky_re_lu_673_layer_call_fn_737105¢
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
K__inference_leaky_re_lu_673_layer_call_and_return_conditional_losses_737110¢
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
": A2dense_743/kernel
:2dense_743/bias
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
*__inference_dense_743_layer_call_fn_737119¢
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
E__inference_dense_743_layer_call_and_return_conditional_losses_737129¢
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
$__inference_signature_wrapper_736082normalization_69_input"
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
':%S2Adam/dense_734/kernel/m
!:S2Adam/dense_734/bias/m
0:.S2$Adam/batch_normalization_665/gamma/m
/:-S2#Adam/batch_normalization_665/beta/m
':%SS2Adam/dense_735/kernel/m
!:S2Adam/dense_735/bias/m
0:.S2$Adam/batch_normalization_666/gamma/m
/:-S2#Adam/batch_normalization_666/beta/m
':%Sa2Adam/dense_736/kernel/m
!:a2Adam/dense_736/bias/m
0:.a2$Adam/batch_normalization_667/gamma/m
/:-a2#Adam/batch_normalization_667/beta/m
':%aa2Adam/dense_737/kernel/m
!:a2Adam/dense_737/bias/m
0:.a2$Adam/batch_normalization_668/gamma/m
/:-a2#Adam/batch_normalization_668/beta/m
':%aA2Adam/dense_738/kernel/m
!:A2Adam/dense_738/bias/m
0:.A2$Adam/batch_normalization_669/gamma/m
/:-A2#Adam/batch_normalization_669/beta/m
':%AA2Adam/dense_739/kernel/m
!:A2Adam/dense_739/bias/m
0:.A2$Adam/batch_normalization_670/gamma/m
/:-A2#Adam/batch_normalization_670/beta/m
':%AA2Adam/dense_740/kernel/m
!:A2Adam/dense_740/bias/m
0:.A2$Adam/batch_normalization_671/gamma/m
/:-A2#Adam/batch_normalization_671/beta/m
':%AA2Adam/dense_741/kernel/m
!:A2Adam/dense_741/bias/m
0:.A2$Adam/batch_normalization_672/gamma/m
/:-A2#Adam/batch_normalization_672/beta/m
':%AA2Adam/dense_742/kernel/m
!:A2Adam/dense_742/bias/m
0:.A2$Adam/batch_normalization_673/gamma/m
/:-A2#Adam/batch_normalization_673/beta/m
':%A2Adam/dense_743/kernel/m
!:2Adam/dense_743/bias/m
':%S2Adam/dense_734/kernel/v
!:S2Adam/dense_734/bias/v
0:.S2$Adam/batch_normalization_665/gamma/v
/:-S2#Adam/batch_normalization_665/beta/v
':%SS2Adam/dense_735/kernel/v
!:S2Adam/dense_735/bias/v
0:.S2$Adam/batch_normalization_666/gamma/v
/:-S2#Adam/batch_normalization_666/beta/v
':%Sa2Adam/dense_736/kernel/v
!:a2Adam/dense_736/bias/v
0:.a2$Adam/batch_normalization_667/gamma/v
/:-a2#Adam/batch_normalization_667/beta/v
':%aa2Adam/dense_737/kernel/v
!:a2Adam/dense_737/bias/v
0:.a2$Adam/batch_normalization_668/gamma/v
/:-a2#Adam/batch_normalization_668/beta/v
':%aA2Adam/dense_738/kernel/v
!:A2Adam/dense_738/bias/v
0:.A2$Adam/batch_normalization_669/gamma/v
/:-A2#Adam/batch_normalization_669/beta/v
':%AA2Adam/dense_739/kernel/v
!:A2Adam/dense_739/bias/v
0:.A2$Adam/batch_normalization_670/gamma/v
/:-A2#Adam/batch_normalization_670/beta/v
':%AA2Adam/dense_740/kernel/v
!:A2Adam/dense_740/bias/v
0:.A2$Adam/batch_normalization_671/gamma/v
/:-A2#Adam/batch_normalization_671/beta/v
':%AA2Adam/dense_741/kernel/v
!:A2Adam/dense_741/bias/v
0:.A2$Adam/batch_normalization_672/gamma/v
/:-A2#Adam/batch_normalization_672/beta/v
':%AA2Adam/dense_742/kernel/v
!:A2Adam/dense_742/bias/v
0:.A2$Adam/batch_normalization_673/gamma/v
/:-A2#Adam/batch_normalization_673/beta/v
':%A2Adam/dense_743/kernel/v
!:2Adam/dense_743/bias/v
	J
Const
J	
Const_1
!__inference__wrapped_model_732993Ú`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøù?¢<
5¢2
0-
normalization_69_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_743# 
	dense_743ÿÿÿÿÿÿÿÿÿf
__inference_adapt_step_736129E-+,:¢7
0¢-
+(¢
 	IteratorSpec 
ª "
 ¹
S__inference_batch_normalization_665_layer_call_and_return_conditional_losses_736194b<9;:3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿS
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿS
 ¹
S__inference_batch_normalization_665_layer_call_and_return_conditional_losses_736228b;<9:3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿS
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿS
 
8__inference_batch_normalization_665_layer_call_fn_736161U<9;:3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿS
p 
ª "ÿÿÿÿÿÿÿÿÿS
8__inference_batch_normalization_665_layer_call_fn_736174U;<9:3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿS
p
ª "ÿÿÿÿÿÿÿÿÿS¹
S__inference_batch_normalization_666_layer_call_and_return_conditional_losses_736303bURTS3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿS
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿS
 ¹
S__inference_batch_normalization_666_layer_call_and_return_conditional_losses_736337bTURS3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿS
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿS
 
8__inference_batch_normalization_666_layer_call_fn_736270UURTS3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿS
p 
ª "ÿÿÿÿÿÿÿÿÿS
8__inference_batch_normalization_666_layer_call_fn_736283UTURS3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿS
p
ª "ÿÿÿÿÿÿÿÿÿS¹
S__inference_batch_normalization_667_layer_call_and_return_conditional_losses_736412bnkml3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 ¹
S__inference_batch_normalization_667_layer_call_and_return_conditional_losses_736446bmnkl3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 
8__inference_batch_normalization_667_layer_call_fn_736379Unkml3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p 
ª "ÿÿÿÿÿÿÿÿÿa
8__inference_batch_normalization_667_layer_call_fn_736392Umnkl3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p
ª "ÿÿÿÿÿÿÿÿÿa½
S__inference_batch_normalization_668_layer_call_and_return_conditional_losses_736521f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 ½
S__inference_batch_normalization_668_layer_call_and_return_conditional_losses_736555f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 
8__inference_batch_normalization_668_layer_call_fn_736488Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p 
ª "ÿÿÿÿÿÿÿÿÿa
8__inference_batch_normalization_668_layer_call_fn_736501Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿa
p
ª "ÿÿÿÿÿÿÿÿÿa½
S__inference_batch_normalization_669_layer_call_and_return_conditional_losses_736630f 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 ½
S__inference_batch_normalization_669_layer_call_and_return_conditional_losses_736664f 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
8__inference_batch_normalization_669_layer_call_fn_736597Y 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "ÿÿÿÿÿÿÿÿÿA
8__inference_batch_normalization_669_layer_call_fn_736610Y 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "ÿÿÿÿÿÿÿÿÿA½
S__inference_batch_normalization_670_layer_call_and_return_conditional_losses_736739f¹¶¸·3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 ½
S__inference_batch_normalization_670_layer_call_and_return_conditional_losses_736773f¸¹¶·3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
8__inference_batch_normalization_670_layer_call_fn_736706Y¹¶¸·3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "ÿÿÿÿÿÿÿÿÿA
8__inference_batch_normalization_670_layer_call_fn_736719Y¸¹¶·3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "ÿÿÿÿÿÿÿÿÿA½
S__inference_batch_normalization_671_layer_call_and_return_conditional_losses_736848fÒÏÑÐ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 ½
S__inference_batch_normalization_671_layer_call_and_return_conditional_losses_736882fÑÒÏÐ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
8__inference_batch_normalization_671_layer_call_fn_736815YÒÏÑÐ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "ÿÿÿÿÿÿÿÿÿA
8__inference_batch_normalization_671_layer_call_fn_736828YÑÒÏÐ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "ÿÿÿÿÿÿÿÿÿA½
S__inference_batch_normalization_672_layer_call_and_return_conditional_losses_736957fëèêé3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 ½
S__inference_batch_normalization_672_layer_call_and_return_conditional_losses_736991fêëèé3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
8__inference_batch_normalization_672_layer_call_fn_736924Yëèêé3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "ÿÿÿÿÿÿÿÿÿA
8__inference_batch_normalization_672_layer_call_fn_736937Yêëèé3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "ÿÿÿÿÿÿÿÿÿA½
S__inference_batch_normalization_673_layer_call_and_return_conditional_losses_737066f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 ½
S__inference_batch_normalization_673_layer_call_and_return_conditional_losses_737100f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
8__inference_batch_normalization_673_layer_call_fn_737033Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p 
ª "ÿÿÿÿÿÿÿÿÿA
8__inference_batch_normalization_673_layer_call_fn_737046Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿA
p
ª "ÿÿÿÿÿÿÿÿÿA¥
E__inference_dense_734_layer_call_and_return_conditional_losses_736148\01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿS
 }
*__inference_dense_734_layer_call_fn_736138O01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿS¥
E__inference_dense_735_layer_call_and_return_conditional_losses_736257\IJ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿS
ª "%¢"

0ÿÿÿÿÿÿÿÿÿS
 }
*__inference_dense_735_layer_call_fn_736247OIJ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿS
ª "ÿÿÿÿÿÿÿÿÿS¥
E__inference_dense_736_layer_call_and_return_conditional_losses_736366\bc/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿS
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 }
*__inference_dense_736_layer_call_fn_736356Obc/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿS
ª "ÿÿÿÿÿÿÿÿÿa¥
E__inference_dense_737_layer_call_and_return_conditional_losses_736475\{|/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 }
*__inference_dense_737_layer_call_fn_736465O{|/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "ÿÿÿÿÿÿÿÿÿa§
E__inference_dense_738_layer_call_and_return_conditional_losses_736584^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
*__inference_dense_738_layer_call_fn_736574Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "ÿÿÿÿÿÿÿÿÿA§
E__inference_dense_739_layer_call_and_return_conditional_losses_736693^­®/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
*__inference_dense_739_layer_call_fn_736683Q­®/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA§
E__inference_dense_740_layer_call_and_return_conditional_losses_736802^ÆÇ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
*__inference_dense_740_layer_call_fn_736792QÆÇ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA§
E__inference_dense_741_layer_call_and_return_conditional_losses_736911^ßà/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
*__inference_dense_741_layer_call_fn_736901Qßà/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA§
E__inference_dense_742_layer_call_and_return_conditional_losses_737020^øù/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
*__inference_dense_742_layer_call_fn_737010Qøù/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA§
E__inference_dense_743_layer_call_and_return_conditional_losses_737129^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_743_layer_call_fn_737119Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_665_layer_call_and_return_conditional_losses_736238X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿS
ª "%¢"

0ÿÿÿÿÿÿÿÿÿS
 
0__inference_leaky_re_lu_665_layer_call_fn_736233K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿS
ª "ÿÿÿÿÿÿÿÿÿS§
K__inference_leaky_re_lu_666_layer_call_and_return_conditional_losses_736347X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿS
ª "%¢"

0ÿÿÿÿÿÿÿÿÿS
 
0__inference_leaky_re_lu_666_layer_call_fn_736342K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿS
ª "ÿÿÿÿÿÿÿÿÿS§
K__inference_leaky_re_lu_667_layer_call_and_return_conditional_losses_736456X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 
0__inference_leaky_re_lu_667_layer_call_fn_736451K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "ÿÿÿÿÿÿÿÿÿa§
K__inference_leaky_re_lu_668_layer_call_and_return_conditional_losses_736565X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "%¢"

0ÿÿÿÿÿÿÿÿÿa
 
0__inference_leaky_re_lu_668_layer_call_fn_736560K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿa
ª "ÿÿÿÿÿÿÿÿÿa§
K__inference_leaky_re_lu_669_layer_call_and_return_conditional_losses_736674X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
0__inference_leaky_re_lu_669_layer_call_fn_736669K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA§
K__inference_leaky_re_lu_670_layer_call_and_return_conditional_losses_736783X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
0__inference_leaky_re_lu_670_layer_call_fn_736778K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA§
K__inference_leaky_re_lu_671_layer_call_and_return_conditional_losses_736892X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
0__inference_leaky_re_lu_671_layer_call_fn_736887K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA§
K__inference_leaky_re_lu_672_layer_call_and_return_conditional_losses_737001X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
0__inference_leaky_re_lu_672_layer_call_fn_736996K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA§
K__inference_leaky_re_lu_673_layer_call_and_return_conditional_losses_737110X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "%¢"

0ÿÿÿÿÿÿÿÿÿA
 
0__inference_leaky_re_lu_673_layer_call_fn_737105K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿA
ª "ÿÿÿÿÿÿÿÿÿA 
I__inference_sequential_69_layer_call_and_return_conditional_losses_734988Ò`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøùG¢D
=¢:
0-
normalization_69_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
  
I__inference_sequential_69_layer_call_and_return_conditional_losses_735139Ò`01;<9:IJTURSbcmnkl{| ­®¸¹¶·ÆÇÑÒÏÐßàêëèéøùG¢D
=¢:
0-
normalization_69_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
I__inference_sequential_69_layer_call_and_return_conditional_losses_735609Â`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøù7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
I__inference_sequential_69_layer_call_and_return_conditional_losses_735959Â`01;<9:IJTURSbcmnkl{| ­®¸¹¶·ÆÇÑÒÏÐßàêëèéøù7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ø
.__inference_sequential_69_layer_call_fn_734169Å`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøùG¢D
=¢:
0-
normalization_69_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿø
.__inference_sequential_69_layer_call_fn_734837Å`01;<9:IJTURSbcmnkl{| ­®¸¹¶·ÆÇÑÒÏÐßàêëèéøùG¢D
=¢:
0-
normalization_69_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿè
.__inference_sequential_69_layer_call_fn_735264µ`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøù7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿè
.__inference_sequential_69_layer_call_fn_735385µ`01;<9:IJTURSbcmnkl{| ­®¸¹¶·ÆÇÑÒÏÐßàêëèéøù7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_signature_wrapper_736082ô`01<9;:IJURTSbcnkml{| ­®¹¶¸·ÆÇÒÏÑÐßàëèêéøùY¢V
¢ 
OªL
J
normalization_69_input0-
normalization_69_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_743# 
	dense_743ÿÿÿÿÿÿÿÿÿ