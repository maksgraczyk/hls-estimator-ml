¤¢
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68áß
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
dense_750/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:.*!
shared_namedense_750/kernel
u
$dense_750/kernel/Read/ReadVariableOpReadVariableOpdense_750/kernel*
_output_shapes

:.*
dtype0
t
dense_750/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*
shared_namedense_750/bias
m
"dense_750/bias/Read/ReadVariableOpReadVariableOpdense_750/bias*
_output_shapes
:.*
dtype0

batch_normalization_676/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*.
shared_namebatch_normalization_676/gamma

1batch_normalization_676/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_676/gamma*
_output_shapes
:.*
dtype0

batch_normalization_676/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*-
shared_namebatch_normalization_676/beta

0batch_normalization_676/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_676/beta*
_output_shapes
:.*
dtype0

#batch_normalization_676/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*4
shared_name%#batch_normalization_676/moving_mean

7batch_normalization_676/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_676/moving_mean*
_output_shapes
:.*
dtype0
¦
'batch_normalization_676/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*8
shared_name)'batch_normalization_676/moving_variance

;batch_normalization_676/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_676/moving_variance*
_output_shapes
:.*
dtype0
|
dense_751/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:..*!
shared_namedense_751/kernel
u
$dense_751/kernel/Read/ReadVariableOpReadVariableOpdense_751/kernel*
_output_shapes

:..*
dtype0
t
dense_751/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*
shared_namedense_751/bias
m
"dense_751/bias/Read/ReadVariableOpReadVariableOpdense_751/bias*
_output_shapes
:.*
dtype0

batch_normalization_677/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*.
shared_namebatch_normalization_677/gamma

1batch_normalization_677/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_677/gamma*
_output_shapes
:.*
dtype0

batch_normalization_677/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*-
shared_namebatch_normalization_677/beta

0batch_normalization_677/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_677/beta*
_output_shapes
:.*
dtype0

#batch_normalization_677/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*4
shared_name%#batch_normalization_677/moving_mean

7batch_normalization_677/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_677/moving_mean*
_output_shapes
:.*
dtype0
¦
'batch_normalization_677/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*8
shared_name)'batch_normalization_677/moving_variance

;batch_normalization_677/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_677/moving_variance*
_output_shapes
:.*
dtype0
|
dense_752/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:.G*!
shared_namedense_752/kernel
u
$dense_752/kernel/Read/ReadVariableOpReadVariableOpdense_752/kernel*
_output_shapes

:.G*
dtype0
t
dense_752/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*
shared_namedense_752/bias
m
"dense_752/bias/Read/ReadVariableOpReadVariableOpdense_752/bias*
_output_shapes
:G*
dtype0

batch_normalization_678/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*.
shared_namebatch_normalization_678/gamma

1batch_normalization_678/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_678/gamma*
_output_shapes
:G*
dtype0

batch_normalization_678/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*-
shared_namebatch_normalization_678/beta

0batch_normalization_678/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_678/beta*
_output_shapes
:G*
dtype0

#batch_normalization_678/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*4
shared_name%#batch_normalization_678/moving_mean

7batch_normalization_678/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_678/moving_mean*
_output_shapes
:G*
dtype0
¦
'batch_normalization_678/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*8
shared_name)'batch_normalization_678/moving_variance

;batch_normalization_678/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_678/moving_variance*
_output_shapes
:G*
dtype0
|
dense_753/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:GG*!
shared_namedense_753/kernel
u
$dense_753/kernel/Read/ReadVariableOpReadVariableOpdense_753/kernel*
_output_shapes

:GG*
dtype0
t
dense_753/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*
shared_namedense_753/bias
m
"dense_753/bias/Read/ReadVariableOpReadVariableOpdense_753/bias*
_output_shapes
:G*
dtype0

batch_normalization_679/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*.
shared_namebatch_normalization_679/gamma

1batch_normalization_679/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_679/gamma*
_output_shapes
:G*
dtype0

batch_normalization_679/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*-
shared_namebatch_normalization_679/beta

0batch_normalization_679/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_679/beta*
_output_shapes
:G*
dtype0

#batch_normalization_679/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*4
shared_name%#batch_normalization_679/moving_mean

7batch_normalization_679/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_679/moving_mean*
_output_shapes
:G*
dtype0
¦
'batch_normalization_679/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*8
shared_name)'batch_normalization_679/moving_variance

;batch_normalization_679/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_679/moving_variance*
_output_shapes
:G*
dtype0
|
dense_754/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Gf*!
shared_namedense_754/kernel
u
$dense_754/kernel/Read/ReadVariableOpReadVariableOpdense_754/kernel*
_output_shapes

:Gf*
dtype0
t
dense_754/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*
shared_namedense_754/bias
m
"dense_754/bias/Read/ReadVariableOpReadVariableOpdense_754/bias*
_output_shapes
:f*
dtype0

batch_normalization_680/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*.
shared_namebatch_normalization_680/gamma

1batch_normalization_680/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_680/gamma*
_output_shapes
:f*
dtype0

batch_normalization_680/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*-
shared_namebatch_normalization_680/beta

0batch_normalization_680/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_680/beta*
_output_shapes
:f*
dtype0

#batch_normalization_680/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*4
shared_name%#batch_normalization_680/moving_mean

7batch_normalization_680/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_680/moving_mean*
_output_shapes
:f*
dtype0
¦
'batch_normalization_680/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*8
shared_name)'batch_normalization_680/moving_variance

;batch_normalization_680/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_680/moving_variance*
_output_shapes
:f*
dtype0
|
dense_755/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:f*!
shared_namedense_755/kernel
u
$dense_755/kernel/Read/ReadVariableOpReadVariableOpdense_755/kernel*
_output_shapes

:f*
dtype0
t
dense_755/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_755/bias
m
"dense_755/bias/Read/ReadVariableOpReadVariableOpdense_755/bias*
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
Adam/dense_750/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:.*(
shared_nameAdam/dense_750/kernel/m

+Adam/dense_750/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_750/kernel/m*
_output_shapes

:.*
dtype0

Adam/dense_750/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*&
shared_nameAdam/dense_750/bias/m
{
)Adam/dense_750/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_750/bias/m*
_output_shapes
:.*
dtype0
 
$Adam/batch_normalization_676/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*5
shared_name&$Adam/batch_normalization_676/gamma/m

8Adam/batch_normalization_676/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_676/gamma/m*
_output_shapes
:.*
dtype0

#Adam/batch_normalization_676/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*4
shared_name%#Adam/batch_normalization_676/beta/m

7Adam/batch_normalization_676/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_676/beta/m*
_output_shapes
:.*
dtype0

Adam/dense_751/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:..*(
shared_nameAdam/dense_751/kernel/m

+Adam/dense_751/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_751/kernel/m*
_output_shapes

:..*
dtype0

Adam/dense_751/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*&
shared_nameAdam/dense_751/bias/m
{
)Adam/dense_751/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_751/bias/m*
_output_shapes
:.*
dtype0
 
$Adam/batch_normalization_677/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*5
shared_name&$Adam/batch_normalization_677/gamma/m

8Adam/batch_normalization_677/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_677/gamma/m*
_output_shapes
:.*
dtype0

#Adam/batch_normalization_677/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*4
shared_name%#Adam/batch_normalization_677/beta/m

7Adam/batch_normalization_677/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_677/beta/m*
_output_shapes
:.*
dtype0

Adam/dense_752/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:.G*(
shared_nameAdam/dense_752/kernel/m

+Adam/dense_752/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_752/kernel/m*
_output_shapes

:.G*
dtype0

Adam/dense_752/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*&
shared_nameAdam/dense_752/bias/m
{
)Adam/dense_752/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_752/bias/m*
_output_shapes
:G*
dtype0
 
$Adam/batch_normalization_678/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*5
shared_name&$Adam/batch_normalization_678/gamma/m

8Adam/batch_normalization_678/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_678/gamma/m*
_output_shapes
:G*
dtype0

#Adam/batch_normalization_678/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*4
shared_name%#Adam/batch_normalization_678/beta/m

7Adam/batch_normalization_678/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_678/beta/m*
_output_shapes
:G*
dtype0

Adam/dense_753/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:GG*(
shared_nameAdam/dense_753/kernel/m

+Adam/dense_753/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_753/kernel/m*
_output_shapes

:GG*
dtype0

Adam/dense_753/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*&
shared_nameAdam/dense_753/bias/m
{
)Adam/dense_753/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_753/bias/m*
_output_shapes
:G*
dtype0
 
$Adam/batch_normalization_679/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*5
shared_name&$Adam/batch_normalization_679/gamma/m

8Adam/batch_normalization_679/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_679/gamma/m*
_output_shapes
:G*
dtype0

#Adam/batch_normalization_679/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*4
shared_name%#Adam/batch_normalization_679/beta/m

7Adam/batch_normalization_679/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_679/beta/m*
_output_shapes
:G*
dtype0

Adam/dense_754/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Gf*(
shared_nameAdam/dense_754/kernel/m

+Adam/dense_754/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_754/kernel/m*
_output_shapes

:Gf*
dtype0

Adam/dense_754/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*&
shared_nameAdam/dense_754/bias/m
{
)Adam/dense_754/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_754/bias/m*
_output_shapes
:f*
dtype0
 
$Adam/batch_normalization_680/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*5
shared_name&$Adam/batch_normalization_680/gamma/m

8Adam/batch_normalization_680/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_680/gamma/m*
_output_shapes
:f*
dtype0

#Adam/batch_normalization_680/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*4
shared_name%#Adam/batch_normalization_680/beta/m

7Adam/batch_normalization_680/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_680/beta/m*
_output_shapes
:f*
dtype0

Adam/dense_755/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:f*(
shared_nameAdam/dense_755/kernel/m

+Adam/dense_755/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_755/kernel/m*
_output_shapes

:f*
dtype0

Adam/dense_755/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_755/bias/m
{
)Adam/dense_755/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_755/bias/m*
_output_shapes
:*
dtype0

Adam/dense_750/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:.*(
shared_nameAdam/dense_750/kernel/v

+Adam/dense_750/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_750/kernel/v*
_output_shapes

:.*
dtype0

Adam/dense_750/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*&
shared_nameAdam/dense_750/bias/v
{
)Adam/dense_750/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_750/bias/v*
_output_shapes
:.*
dtype0
 
$Adam/batch_normalization_676/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*5
shared_name&$Adam/batch_normalization_676/gamma/v

8Adam/batch_normalization_676/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_676/gamma/v*
_output_shapes
:.*
dtype0

#Adam/batch_normalization_676/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*4
shared_name%#Adam/batch_normalization_676/beta/v

7Adam/batch_normalization_676/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_676/beta/v*
_output_shapes
:.*
dtype0

Adam/dense_751/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:..*(
shared_nameAdam/dense_751/kernel/v

+Adam/dense_751/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_751/kernel/v*
_output_shapes

:..*
dtype0

Adam/dense_751/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*&
shared_nameAdam/dense_751/bias/v
{
)Adam/dense_751/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_751/bias/v*
_output_shapes
:.*
dtype0
 
$Adam/batch_normalization_677/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*5
shared_name&$Adam/batch_normalization_677/gamma/v

8Adam/batch_normalization_677/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_677/gamma/v*
_output_shapes
:.*
dtype0

#Adam/batch_normalization_677/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*4
shared_name%#Adam/batch_normalization_677/beta/v

7Adam/batch_normalization_677/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_677/beta/v*
_output_shapes
:.*
dtype0

Adam/dense_752/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:.G*(
shared_nameAdam/dense_752/kernel/v

+Adam/dense_752/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_752/kernel/v*
_output_shapes

:.G*
dtype0

Adam/dense_752/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*&
shared_nameAdam/dense_752/bias/v
{
)Adam/dense_752/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_752/bias/v*
_output_shapes
:G*
dtype0
 
$Adam/batch_normalization_678/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*5
shared_name&$Adam/batch_normalization_678/gamma/v

8Adam/batch_normalization_678/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_678/gamma/v*
_output_shapes
:G*
dtype0

#Adam/batch_normalization_678/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*4
shared_name%#Adam/batch_normalization_678/beta/v

7Adam/batch_normalization_678/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_678/beta/v*
_output_shapes
:G*
dtype0

Adam/dense_753/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:GG*(
shared_nameAdam/dense_753/kernel/v

+Adam/dense_753/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_753/kernel/v*
_output_shapes

:GG*
dtype0

Adam/dense_753/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*&
shared_nameAdam/dense_753/bias/v
{
)Adam/dense_753/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_753/bias/v*
_output_shapes
:G*
dtype0
 
$Adam/batch_normalization_679/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*5
shared_name&$Adam/batch_normalization_679/gamma/v

8Adam/batch_normalization_679/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_679/gamma/v*
_output_shapes
:G*
dtype0

#Adam/batch_normalization_679/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*4
shared_name%#Adam/batch_normalization_679/beta/v

7Adam/batch_normalization_679/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_679/beta/v*
_output_shapes
:G*
dtype0

Adam/dense_754/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Gf*(
shared_nameAdam/dense_754/kernel/v

+Adam/dense_754/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_754/kernel/v*
_output_shapes

:Gf*
dtype0

Adam/dense_754/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*&
shared_nameAdam/dense_754/bias/v
{
)Adam/dense_754/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_754/bias/v*
_output_shapes
:f*
dtype0
 
$Adam/batch_normalization_680/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*5
shared_name&$Adam/batch_normalization_680/gamma/v

8Adam/batch_normalization_680/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_680/gamma/v*
_output_shapes
:f*
dtype0

#Adam/batch_normalization_680/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*4
shared_name%#Adam/batch_normalization_680/beta/v

7Adam/batch_normalization_680/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_680/beta/v*
_output_shapes
:f*
dtype0

Adam/dense_755/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:f*(
shared_nameAdam/dense_755/kernel/v

+Adam/dense_755/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_755/kernel/v*
_output_shapes

:f*
dtype0

Adam/dense_755/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_755/bias/v
{
)Adam/dense_755/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_755/bias/v*
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
valueB"5sEpÍvE ÀB

NoOpNoOp
Â©
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*ú¨
valueï¨Bë¨ Bã¨
ê
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
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¾

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
 variance
 adapt_variance
	!count
"	keras_api
#_adapt_function*
¦

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses*
Õ
,axis
	-gamma
.beta
/moving_mean
0moving_variance
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses*

7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses* 
¦

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses*
Õ
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses*

P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses* 
¦

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses*
Õ
^axis
	_gamma
`beta
amoving_mean
bmoving_variance
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses*

i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses* 
¦

okernel
pbias
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses*
×
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses* 
®
¡kernel
	¢bias
£	variables
¤trainable_variables
¥regularization_losses
¦	keras_api
§__call__
+¨&call_and_return_all_conditional_losses*
ù
	©iter
ªbeta_1
«beta_2

¬decay$m%m-m.m=m>mFmGmVmWm_m`mompmxmym	m	m	m	m	¡m	¢m$v%v-v .v¡=v¢>v£Fv¤Gv¥Vv¦Wv§_v¨`v©ovªpv«xv¬yv­	v®	v¯	v°	v±	¡v²	¢v³*

0
 1
!2
$3
%4
-5
.6
/7
08
=9
>10
F11
G12
H13
I14
V15
W16
_17
`18
a19
b20
o21
p22
x23
y24
z25
{26
27
28
29
30
31
32
¡33
¢34*
°
$0
%1
-2
.3
=4
>5
F6
G7
V8
W9
_10
`11
o12
p13
x14
y15
16
17
18
19
¡20
¢21*
* 
µ
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

²serving_default* 
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
VARIABLE_VALUEdense_750/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_750/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 

³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_676/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_676/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_676/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_676/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
-0
.1
/2
03*

-0
.1*
* 

¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_751/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_751/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

=0
>1*

=0
>1*
* 

Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_677/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_677/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_677/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_677/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
F0
G1
H2
I3*

F0
G1*
* 

Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_752/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_752/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

V0
W1*

V0
W1*
* 

Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_678/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_678/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_678/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_678/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
_0
`1
a2
b3*

_0
`1*
* 

Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_753/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_753/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

o0
p1*

o0
p1*
* 

ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_679/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_679/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_679/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_679/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
x0
y1
z2
{3*

x0
y1*
* 

ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_754/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_754/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_680/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_680/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_680/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_680/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ùnon_trainable_variables
úlayers
ûmetrics
 ülayer_regularization_losses
ýlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_755/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_755/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

¡0
¢1*

¡0
¢1*
* 

þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
£	variables
¤trainable_variables
¥regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses*
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
d
0
 1
!2
/3
04
H5
I6
a7
b8
z9
{10
11
12*

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
16*

0*
* 
* 
* 
* 
* 
* 
* 
* 

/0
01*
* 
* 
* 
* 
* 
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
H0
I1*
* 
* 
* 
* 
* 
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
a0
b1*
* 
* 
* 
* 
* 
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
z0
{1*
* 
* 
* 
* 
* 
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
0
1*
* 
* 
* 
* 
* 
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

total

count
	variables
	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
}
VARIABLE_VALUEAdam/dense_750/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_750/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_676/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_676/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_751/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_751/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_677/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_677/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_752/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_752/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_678/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_678/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_753/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_753/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_679/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_679/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_754/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_754/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_680/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_680/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_755/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_755/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_750/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_750/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_676/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_676/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_751/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_751/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_677/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_677/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_752/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_752/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_678/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_678/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_753/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_753/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_679/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_679/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_754/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_754/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_680/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_680/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_755/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_755/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

&serving_default_normalization_74_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ


StatefulPartitionedCallStatefulPartitionedCall&serving_default_normalization_74_inputConstConst_1dense_750/kerneldense_750/bias'batch_normalization_676/moving_variancebatch_normalization_676/gamma#batch_normalization_676/moving_meanbatch_normalization_676/betadense_751/kerneldense_751/bias'batch_normalization_677/moving_variancebatch_normalization_677/gamma#batch_normalization_677/moving_meanbatch_normalization_677/betadense_752/kerneldense_752/bias'batch_normalization_678/moving_variancebatch_normalization_678/gamma#batch_normalization_678/moving_meanbatch_normalization_678/betadense_753/kerneldense_753/bias'batch_normalization_679/moving_variancebatch_normalization_679/gamma#batch_normalization_679/moving_meanbatch_normalization_679/betadense_754/kerneldense_754/bias'batch_normalization_680/moving_variancebatch_normalization_680/gamma#batch_normalization_680/moving_meanbatch_normalization_680/betadense_755/kerneldense_755/bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 !"*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_751382
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
"
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp$dense_750/kernel/Read/ReadVariableOp"dense_750/bias/Read/ReadVariableOp1batch_normalization_676/gamma/Read/ReadVariableOp0batch_normalization_676/beta/Read/ReadVariableOp7batch_normalization_676/moving_mean/Read/ReadVariableOp;batch_normalization_676/moving_variance/Read/ReadVariableOp$dense_751/kernel/Read/ReadVariableOp"dense_751/bias/Read/ReadVariableOp1batch_normalization_677/gamma/Read/ReadVariableOp0batch_normalization_677/beta/Read/ReadVariableOp7batch_normalization_677/moving_mean/Read/ReadVariableOp;batch_normalization_677/moving_variance/Read/ReadVariableOp$dense_752/kernel/Read/ReadVariableOp"dense_752/bias/Read/ReadVariableOp1batch_normalization_678/gamma/Read/ReadVariableOp0batch_normalization_678/beta/Read/ReadVariableOp7batch_normalization_678/moving_mean/Read/ReadVariableOp;batch_normalization_678/moving_variance/Read/ReadVariableOp$dense_753/kernel/Read/ReadVariableOp"dense_753/bias/Read/ReadVariableOp1batch_normalization_679/gamma/Read/ReadVariableOp0batch_normalization_679/beta/Read/ReadVariableOp7batch_normalization_679/moving_mean/Read/ReadVariableOp;batch_normalization_679/moving_variance/Read/ReadVariableOp$dense_754/kernel/Read/ReadVariableOp"dense_754/bias/Read/ReadVariableOp1batch_normalization_680/gamma/Read/ReadVariableOp0batch_normalization_680/beta/Read/ReadVariableOp7batch_normalization_680/moving_mean/Read/ReadVariableOp;batch_normalization_680/moving_variance/Read/ReadVariableOp$dense_755/kernel/Read/ReadVariableOp"dense_755/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_750/kernel/m/Read/ReadVariableOp)Adam/dense_750/bias/m/Read/ReadVariableOp8Adam/batch_normalization_676/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_676/beta/m/Read/ReadVariableOp+Adam/dense_751/kernel/m/Read/ReadVariableOp)Adam/dense_751/bias/m/Read/ReadVariableOp8Adam/batch_normalization_677/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_677/beta/m/Read/ReadVariableOp+Adam/dense_752/kernel/m/Read/ReadVariableOp)Adam/dense_752/bias/m/Read/ReadVariableOp8Adam/batch_normalization_678/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_678/beta/m/Read/ReadVariableOp+Adam/dense_753/kernel/m/Read/ReadVariableOp)Adam/dense_753/bias/m/Read/ReadVariableOp8Adam/batch_normalization_679/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_679/beta/m/Read/ReadVariableOp+Adam/dense_754/kernel/m/Read/ReadVariableOp)Adam/dense_754/bias/m/Read/ReadVariableOp8Adam/batch_normalization_680/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_680/beta/m/Read/ReadVariableOp+Adam/dense_755/kernel/m/Read/ReadVariableOp)Adam/dense_755/bias/m/Read/ReadVariableOp+Adam/dense_750/kernel/v/Read/ReadVariableOp)Adam/dense_750/bias/v/Read/ReadVariableOp8Adam/batch_normalization_676/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_676/beta/v/Read/ReadVariableOp+Adam/dense_751/kernel/v/Read/ReadVariableOp)Adam/dense_751/bias/v/Read/ReadVariableOp8Adam/batch_normalization_677/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_677/beta/v/Read/ReadVariableOp+Adam/dense_752/kernel/v/Read/ReadVariableOp)Adam/dense_752/bias/v/Read/ReadVariableOp8Adam/batch_normalization_678/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_678/beta/v/Read/ReadVariableOp+Adam/dense_753/kernel/v/Read/ReadVariableOp)Adam/dense_753/bias/v/Read/ReadVariableOp8Adam/batch_normalization_679/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_679/beta/v/Read/ReadVariableOp+Adam/dense_754/kernel/v/Read/ReadVariableOp)Adam/dense_754/bias/v/Read/ReadVariableOp8Adam/batch_normalization_680/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_680/beta/v/Read/ReadVariableOp+Adam/dense_755/kernel/v/Read/ReadVariableOp)Adam/dense_755/bias/v/Read/ReadVariableOpConst_2*b
Tin[
Y2W		*
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
__inference__traced_save_752273
ô
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_750/kerneldense_750/biasbatch_normalization_676/gammabatch_normalization_676/beta#batch_normalization_676/moving_mean'batch_normalization_676/moving_variancedense_751/kerneldense_751/biasbatch_normalization_677/gammabatch_normalization_677/beta#batch_normalization_677/moving_mean'batch_normalization_677/moving_variancedense_752/kerneldense_752/biasbatch_normalization_678/gammabatch_normalization_678/beta#batch_normalization_678/moving_mean'batch_normalization_678/moving_variancedense_753/kerneldense_753/biasbatch_normalization_679/gammabatch_normalization_679/beta#batch_normalization_679/moving_mean'batch_normalization_679/moving_variancedense_754/kerneldense_754/biasbatch_normalization_680/gammabatch_normalization_680/beta#batch_normalization_680/moving_mean'batch_normalization_680/moving_variancedense_755/kerneldense_755/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount_1Adam/dense_750/kernel/mAdam/dense_750/bias/m$Adam/batch_normalization_676/gamma/m#Adam/batch_normalization_676/beta/mAdam/dense_751/kernel/mAdam/dense_751/bias/m$Adam/batch_normalization_677/gamma/m#Adam/batch_normalization_677/beta/mAdam/dense_752/kernel/mAdam/dense_752/bias/m$Adam/batch_normalization_678/gamma/m#Adam/batch_normalization_678/beta/mAdam/dense_753/kernel/mAdam/dense_753/bias/m$Adam/batch_normalization_679/gamma/m#Adam/batch_normalization_679/beta/mAdam/dense_754/kernel/mAdam/dense_754/bias/m$Adam/batch_normalization_680/gamma/m#Adam/batch_normalization_680/beta/mAdam/dense_755/kernel/mAdam/dense_755/bias/mAdam/dense_750/kernel/vAdam/dense_750/bias/v$Adam/batch_normalization_676/gamma/v#Adam/batch_normalization_676/beta/vAdam/dense_751/kernel/vAdam/dense_751/bias/v$Adam/batch_normalization_677/gamma/v#Adam/batch_normalization_677/beta/vAdam/dense_752/kernel/vAdam/dense_752/bias/v$Adam/batch_normalization_678/gamma/v#Adam/batch_normalization_678/beta/vAdam/dense_753/kernel/vAdam/dense_753/bias/v$Adam/batch_normalization_679/gamma/v#Adam/batch_normalization_679/beta/vAdam/dense_754/kernel/vAdam/dense_754/bias/v$Adam/batch_normalization_680/gamma/v#Adam/batch_normalization_680/beta/vAdam/dense_755/kernel/vAdam/dense_755/bias/v*a
TinZ
X2V*
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
"__inference__traced_restore_752538ð¹
ºå
%
!__inference__wrapped_model_749569
normalization_74_input(
$sequential_74_normalization_74_sub_y)
%sequential_74_normalization_74_sqrt_xH
6sequential_74_dense_750_matmul_readvariableop_resource:.E
7sequential_74_dense_750_biasadd_readvariableop_resource:.U
Gsequential_74_batch_normalization_676_batchnorm_readvariableop_resource:.Y
Ksequential_74_batch_normalization_676_batchnorm_mul_readvariableop_resource:.W
Isequential_74_batch_normalization_676_batchnorm_readvariableop_1_resource:.W
Isequential_74_batch_normalization_676_batchnorm_readvariableop_2_resource:.H
6sequential_74_dense_751_matmul_readvariableop_resource:..E
7sequential_74_dense_751_biasadd_readvariableop_resource:.U
Gsequential_74_batch_normalization_677_batchnorm_readvariableop_resource:.Y
Ksequential_74_batch_normalization_677_batchnorm_mul_readvariableop_resource:.W
Isequential_74_batch_normalization_677_batchnorm_readvariableop_1_resource:.W
Isequential_74_batch_normalization_677_batchnorm_readvariableop_2_resource:.H
6sequential_74_dense_752_matmul_readvariableop_resource:.GE
7sequential_74_dense_752_biasadd_readvariableop_resource:GU
Gsequential_74_batch_normalization_678_batchnorm_readvariableop_resource:GY
Ksequential_74_batch_normalization_678_batchnorm_mul_readvariableop_resource:GW
Isequential_74_batch_normalization_678_batchnorm_readvariableop_1_resource:GW
Isequential_74_batch_normalization_678_batchnorm_readvariableop_2_resource:GH
6sequential_74_dense_753_matmul_readvariableop_resource:GGE
7sequential_74_dense_753_biasadd_readvariableop_resource:GU
Gsequential_74_batch_normalization_679_batchnorm_readvariableop_resource:GY
Ksequential_74_batch_normalization_679_batchnorm_mul_readvariableop_resource:GW
Isequential_74_batch_normalization_679_batchnorm_readvariableop_1_resource:GW
Isequential_74_batch_normalization_679_batchnorm_readvariableop_2_resource:GH
6sequential_74_dense_754_matmul_readvariableop_resource:GfE
7sequential_74_dense_754_biasadd_readvariableop_resource:fU
Gsequential_74_batch_normalization_680_batchnorm_readvariableop_resource:fY
Ksequential_74_batch_normalization_680_batchnorm_mul_readvariableop_resource:fW
Isequential_74_batch_normalization_680_batchnorm_readvariableop_1_resource:fW
Isequential_74_batch_normalization_680_batchnorm_readvariableop_2_resource:fH
6sequential_74_dense_755_matmul_readvariableop_resource:fE
7sequential_74_dense_755_biasadd_readvariableop_resource:
identity¢>sequential_74/batch_normalization_676/batchnorm/ReadVariableOp¢@sequential_74/batch_normalization_676/batchnorm/ReadVariableOp_1¢@sequential_74/batch_normalization_676/batchnorm/ReadVariableOp_2¢Bsequential_74/batch_normalization_676/batchnorm/mul/ReadVariableOp¢>sequential_74/batch_normalization_677/batchnorm/ReadVariableOp¢@sequential_74/batch_normalization_677/batchnorm/ReadVariableOp_1¢@sequential_74/batch_normalization_677/batchnorm/ReadVariableOp_2¢Bsequential_74/batch_normalization_677/batchnorm/mul/ReadVariableOp¢>sequential_74/batch_normalization_678/batchnorm/ReadVariableOp¢@sequential_74/batch_normalization_678/batchnorm/ReadVariableOp_1¢@sequential_74/batch_normalization_678/batchnorm/ReadVariableOp_2¢Bsequential_74/batch_normalization_678/batchnorm/mul/ReadVariableOp¢>sequential_74/batch_normalization_679/batchnorm/ReadVariableOp¢@sequential_74/batch_normalization_679/batchnorm/ReadVariableOp_1¢@sequential_74/batch_normalization_679/batchnorm/ReadVariableOp_2¢Bsequential_74/batch_normalization_679/batchnorm/mul/ReadVariableOp¢>sequential_74/batch_normalization_680/batchnorm/ReadVariableOp¢@sequential_74/batch_normalization_680/batchnorm/ReadVariableOp_1¢@sequential_74/batch_normalization_680/batchnorm/ReadVariableOp_2¢Bsequential_74/batch_normalization_680/batchnorm/mul/ReadVariableOp¢.sequential_74/dense_750/BiasAdd/ReadVariableOp¢-sequential_74/dense_750/MatMul/ReadVariableOp¢.sequential_74/dense_751/BiasAdd/ReadVariableOp¢-sequential_74/dense_751/MatMul/ReadVariableOp¢.sequential_74/dense_752/BiasAdd/ReadVariableOp¢-sequential_74/dense_752/MatMul/ReadVariableOp¢.sequential_74/dense_753/BiasAdd/ReadVariableOp¢-sequential_74/dense_753/MatMul/ReadVariableOp¢.sequential_74/dense_754/BiasAdd/ReadVariableOp¢-sequential_74/dense_754/MatMul/ReadVariableOp¢.sequential_74/dense_755/BiasAdd/ReadVariableOp¢-sequential_74/dense_755/MatMul/ReadVariableOp
"sequential_74/normalization_74/subSubnormalization_74_input$sequential_74_normalization_74_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
#sequential_74/normalization_74/SqrtSqrt%sequential_74_normalization_74_sqrt_x*
T0*
_output_shapes

:m
(sequential_74/normalization_74/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¶
&sequential_74/normalization_74/MaximumMaximum'sequential_74/normalization_74/Sqrt:y:01sequential_74/normalization_74/Maximum/y:output:0*
T0*
_output_shapes

:·
&sequential_74/normalization_74/truedivRealDiv&sequential_74/normalization_74/sub:z:0*sequential_74/normalization_74/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_74/dense_750/MatMul/ReadVariableOpReadVariableOp6sequential_74_dense_750_matmul_readvariableop_resource*
_output_shapes

:.*
dtype0½
sequential_74/dense_750/MatMulMatMul*sequential_74/normalization_74/truediv:z:05sequential_74/dense_750/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.¢
.sequential_74/dense_750/BiasAdd/ReadVariableOpReadVariableOp7sequential_74_dense_750_biasadd_readvariableop_resource*
_output_shapes
:.*
dtype0¾
sequential_74/dense_750/BiasAddBiasAdd(sequential_74/dense_750/MatMul:product:06sequential_74/dense_750/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.Â
>sequential_74/batch_normalization_676/batchnorm/ReadVariableOpReadVariableOpGsequential_74_batch_normalization_676_batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0z
5sequential_74/batch_normalization_676/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_74/batch_normalization_676/batchnorm/addAddV2Fsequential_74/batch_normalization_676/batchnorm/ReadVariableOp:value:0>sequential_74/batch_normalization_676/batchnorm/add/y:output:0*
T0*
_output_shapes
:.
5sequential_74/batch_normalization_676/batchnorm/RsqrtRsqrt7sequential_74/batch_normalization_676/batchnorm/add:z:0*
T0*
_output_shapes
:.Ê
Bsequential_74/batch_normalization_676/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_74_batch_normalization_676_batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0æ
3sequential_74/batch_normalization_676/batchnorm/mulMul9sequential_74/batch_normalization_676/batchnorm/Rsqrt:y:0Jsequential_74/batch_normalization_676/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.Ñ
5sequential_74/batch_normalization_676/batchnorm/mul_1Mul(sequential_74/dense_750/BiasAdd:output:07sequential_74/batch_normalization_676/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.Æ
@sequential_74/batch_normalization_676/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_74_batch_normalization_676_batchnorm_readvariableop_1_resource*
_output_shapes
:.*
dtype0ä
5sequential_74/batch_normalization_676/batchnorm/mul_2MulHsequential_74/batch_normalization_676/batchnorm/ReadVariableOp_1:value:07sequential_74/batch_normalization_676/batchnorm/mul:z:0*
T0*
_output_shapes
:.Æ
@sequential_74/batch_normalization_676/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_74_batch_normalization_676_batchnorm_readvariableop_2_resource*
_output_shapes
:.*
dtype0ä
3sequential_74/batch_normalization_676/batchnorm/subSubHsequential_74/batch_normalization_676/batchnorm/ReadVariableOp_2:value:09sequential_74/batch_normalization_676/batchnorm/mul_2:z:0*
T0*
_output_shapes
:.ä
5sequential_74/batch_normalization_676/batchnorm/add_1AddV29sequential_74/batch_normalization_676/batchnorm/mul_1:z:07sequential_74/batch_normalization_676/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.¨
'sequential_74/leaky_re_lu_676/LeakyRelu	LeakyRelu9sequential_74/batch_normalization_676/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
alpha%>¤
-sequential_74/dense_751/MatMul/ReadVariableOpReadVariableOp6sequential_74_dense_751_matmul_readvariableop_resource*
_output_shapes

:..*
dtype0È
sequential_74/dense_751/MatMulMatMul5sequential_74/leaky_re_lu_676/LeakyRelu:activations:05sequential_74/dense_751/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.¢
.sequential_74/dense_751/BiasAdd/ReadVariableOpReadVariableOp7sequential_74_dense_751_biasadd_readvariableop_resource*
_output_shapes
:.*
dtype0¾
sequential_74/dense_751/BiasAddBiasAdd(sequential_74/dense_751/MatMul:product:06sequential_74/dense_751/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.Â
>sequential_74/batch_normalization_677/batchnorm/ReadVariableOpReadVariableOpGsequential_74_batch_normalization_677_batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0z
5sequential_74/batch_normalization_677/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_74/batch_normalization_677/batchnorm/addAddV2Fsequential_74/batch_normalization_677/batchnorm/ReadVariableOp:value:0>sequential_74/batch_normalization_677/batchnorm/add/y:output:0*
T0*
_output_shapes
:.
5sequential_74/batch_normalization_677/batchnorm/RsqrtRsqrt7sequential_74/batch_normalization_677/batchnorm/add:z:0*
T0*
_output_shapes
:.Ê
Bsequential_74/batch_normalization_677/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_74_batch_normalization_677_batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0æ
3sequential_74/batch_normalization_677/batchnorm/mulMul9sequential_74/batch_normalization_677/batchnorm/Rsqrt:y:0Jsequential_74/batch_normalization_677/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.Ñ
5sequential_74/batch_normalization_677/batchnorm/mul_1Mul(sequential_74/dense_751/BiasAdd:output:07sequential_74/batch_normalization_677/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.Æ
@sequential_74/batch_normalization_677/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_74_batch_normalization_677_batchnorm_readvariableop_1_resource*
_output_shapes
:.*
dtype0ä
5sequential_74/batch_normalization_677/batchnorm/mul_2MulHsequential_74/batch_normalization_677/batchnorm/ReadVariableOp_1:value:07sequential_74/batch_normalization_677/batchnorm/mul:z:0*
T0*
_output_shapes
:.Æ
@sequential_74/batch_normalization_677/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_74_batch_normalization_677_batchnorm_readvariableop_2_resource*
_output_shapes
:.*
dtype0ä
3sequential_74/batch_normalization_677/batchnorm/subSubHsequential_74/batch_normalization_677/batchnorm/ReadVariableOp_2:value:09sequential_74/batch_normalization_677/batchnorm/mul_2:z:0*
T0*
_output_shapes
:.ä
5sequential_74/batch_normalization_677/batchnorm/add_1AddV29sequential_74/batch_normalization_677/batchnorm/mul_1:z:07sequential_74/batch_normalization_677/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.¨
'sequential_74/leaky_re_lu_677/LeakyRelu	LeakyRelu9sequential_74/batch_normalization_677/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
alpha%>¤
-sequential_74/dense_752/MatMul/ReadVariableOpReadVariableOp6sequential_74_dense_752_matmul_readvariableop_resource*
_output_shapes

:.G*
dtype0È
sequential_74/dense_752/MatMulMatMul5sequential_74/leaky_re_lu_677/LeakyRelu:activations:05sequential_74/dense_752/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG¢
.sequential_74/dense_752/BiasAdd/ReadVariableOpReadVariableOp7sequential_74_dense_752_biasadd_readvariableop_resource*
_output_shapes
:G*
dtype0¾
sequential_74/dense_752/BiasAddBiasAdd(sequential_74/dense_752/MatMul:product:06sequential_74/dense_752/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGÂ
>sequential_74/batch_normalization_678/batchnorm/ReadVariableOpReadVariableOpGsequential_74_batch_normalization_678_batchnorm_readvariableop_resource*
_output_shapes
:G*
dtype0z
5sequential_74/batch_normalization_678/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_74/batch_normalization_678/batchnorm/addAddV2Fsequential_74/batch_normalization_678/batchnorm/ReadVariableOp:value:0>sequential_74/batch_normalization_678/batchnorm/add/y:output:0*
T0*
_output_shapes
:G
5sequential_74/batch_normalization_678/batchnorm/RsqrtRsqrt7sequential_74/batch_normalization_678/batchnorm/add:z:0*
T0*
_output_shapes
:GÊ
Bsequential_74/batch_normalization_678/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_74_batch_normalization_678_batchnorm_mul_readvariableop_resource*
_output_shapes
:G*
dtype0æ
3sequential_74/batch_normalization_678/batchnorm/mulMul9sequential_74/batch_normalization_678/batchnorm/Rsqrt:y:0Jsequential_74/batch_normalization_678/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:GÑ
5sequential_74/batch_normalization_678/batchnorm/mul_1Mul(sequential_74/dense_752/BiasAdd:output:07sequential_74/batch_normalization_678/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGÆ
@sequential_74/batch_normalization_678/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_74_batch_normalization_678_batchnorm_readvariableop_1_resource*
_output_shapes
:G*
dtype0ä
5sequential_74/batch_normalization_678/batchnorm/mul_2MulHsequential_74/batch_normalization_678/batchnorm/ReadVariableOp_1:value:07sequential_74/batch_normalization_678/batchnorm/mul:z:0*
T0*
_output_shapes
:GÆ
@sequential_74/batch_normalization_678/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_74_batch_normalization_678_batchnorm_readvariableop_2_resource*
_output_shapes
:G*
dtype0ä
3sequential_74/batch_normalization_678/batchnorm/subSubHsequential_74/batch_normalization_678/batchnorm/ReadVariableOp_2:value:09sequential_74/batch_normalization_678/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Gä
5sequential_74/batch_normalization_678/batchnorm/add_1AddV29sequential_74/batch_normalization_678/batchnorm/mul_1:z:07sequential_74/batch_normalization_678/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG¨
'sequential_74/leaky_re_lu_678/LeakyRelu	LeakyRelu9sequential_74/batch_normalization_678/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
alpha%>¤
-sequential_74/dense_753/MatMul/ReadVariableOpReadVariableOp6sequential_74_dense_753_matmul_readvariableop_resource*
_output_shapes

:GG*
dtype0È
sequential_74/dense_753/MatMulMatMul5sequential_74/leaky_re_lu_678/LeakyRelu:activations:05sequential_74/dense_753/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG¢
.sequential_74/dense_753/BiasAdd/ReadVariableOpReadVariableOp7sequential_74_dense_753_biasadd_readvariableop_resource*
_output_shapes
:G*
dtype0¾
sequential_74/dense_753/BiasAddBiasAdd(sequential_74/dense_753/MatMul:product:06sequential_74/dense_753/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGÂ
>sequential_74/batch_normalization_679/batchnorm/ReadVariableOpReadVariableOpGsequential_74_batch_normalization_679_batchnorm_readvariableop_resource*
_output_shapes
:G*
dtype0z
5sequential_74/batch_normalization_679/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_74/batch_normalization_679/batchnorm/addAddV2Fsequential_74/batch_normalization_679/batchnorm/ReadVariableOp:value:0>sequential_74/batch_normalization_679/batchnorm/add/y:output:0*
T0*
_output_shapes
:G
5sequential_74/batch_normalization_679/batchnorm/RsqrtRsqrt7sequential_74/batch_normalization_679/batchnorm/add:z:0*
T0*
_output_shapes
:GÊ
Bsequential_74/batch_normalization_679/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_74_batch_normalization_679_batchnorm_mul_readvariableop_resource*
_output_shapes
:G*
dtype0æ
3sequential_74/batch_normalization_679/batchnorm/mulMul9sequential_74/batch_normalization_679/batchnorm/Rsqrt:y:0Jsequential_74/batch_normalization_679/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:GÑ
5sequential_74/batch_normalization_679/batchnorm/mul_1Mul(sequential_74/dense_753/BiasAdd:output:07sequential_74/batch_normalization_679/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGÆ
@sequential_74/batch_normalization_679/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_74_batch_normalization_679_batchnorm_readvariableop_1_resource*
_output_shapes
:G*
dtype0ä
5sequential_74/batch_normalization_679/batchnorm/mul_2MulHsequential_74/batch_normalization_679/batchnorm/ReadVariableOp_1:value:07sequential_74/batch_normalization_679/batchnorm/mul:z:0*
T0*
_output_shapes
:GÆ
@sequential_74/batch_normalization_679/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_74_batch_normalization_679_batchnorm_readvariableop_2_resource*
_output_shapes
:G*
dtype0ä
3sequential_74/batch_normalization_679/batchnorm/subSubHsequential_74/batch_normalization_679/batchnorm/ReadVariableOp_2:value:09sequential_74/batch_normalization_679/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Gä
5sequential_74/batch_normalization_679/batchnorm/add_1AddV29sequential_74/batch_normalization_679/batchnorm/mul_1:z:07sequential_74/batch_normalization_679/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG¨
'sequential_74/leaky_re_lu_679/LeakyRelu	LeakyRelu9sequential_74/batch_normalization_679/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
alpha%>¤
-sequential_74/dense_754/MatMul/ReadVariableOpReadVariableOp6sequential_74_dense_754_matmul_readvariableop_resource*
_output_shapes

:Gf*
dtype0È
sequential_74/dense_754/MatMulMatMul5sequential_74/leaky_re_lu_679/LeakyRelu:activations:05sequential_74/dense_754/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf¢
.sequential_74/dense_754/BiasAdd/ReadVariableOpReadVariableOp7sequential_74_dense_754_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0¾
sequential_74/dense_754/BiasAddBiasAdd(sequential_74/dense_754/MatMul:product:06sequential_74/dense_754/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfÂ
>sequential_74/batch_normalization_680/batchnorm/ReadVariableOpReadVariableOpGsequential_74_batch_normalization_680_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0z
5sequential_74/batch_normalization_680/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_74/batch_normalization_680/batchnorm/addAddV2Fsequential_74/batch_normalization_680/batchnorm/ReadVariableOp:value:0>sequential_74/batch_normalization_680/batchnorm/add/y:output:0*
T0*
_output_shapes
:f
5sequential_74/batch_normalization_680/batchnorm/RsqrtRsqrt7sequential_74/batch_normalization_680/batchnorm/add:z:0*
T0*
_output_shapes
:fÊ
Bsequential_74/batch_normalization_680/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_74_batch_normalization_680_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0æ
3sequential_74/batch_normalization_680/batchnorm/mulMul9sequential_74/batch_normalization_680/batchnorm/Rsqrt:y:0Jsequential_74/batch_normalization_680/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:fÑ
5sequential_74/batch_normalization_680/batchnorm/mul_1Mul(sequential_74/dense_754/BiasAdd:output:07sequential_74/batch_normalization_680/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfÆ
@sequential_74/batch_normalization_680/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_74_batch_normalization_680_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0ä
5sequential_74/batch_normalization_680/batchnorm/mul_2MulHsequential_74/batch_normalization_680/batchnorm/ReadVariableOp_1:value:07sequential_74/batch_normalization_680/batchnorm/mul:z:0*
T0*
_output_shapes
:fÆ
@sequential_74/batch_normalization_680/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_74_batch_normalization_680_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0ä
3sequential_74/batch_normalization_680/batchnorm/subSubHsequential_74/batch_normalization_680/batchnorm/ReadVariableOp_2:value:09sequential_74/batch_normalization_680/batchnorm/mul_2:z:0*
T0*
_output_shapes
:fä
5sequential_74/batch_normalization_680/batchnorm/add_1AddV29sequential_74/batch_normalization_680/batchnorm/mul_1:z:07sequential_74/batch_normalization_680/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf¨
'sequential_74/leaky_re_lu_680/LeakyRelu	LeakyRelu9sequential_74/batch_normalization_680/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*
alpha%>¤
-sequential_74/dense_755/MatMul/ReadVariableOpReadVariableOp6sequential_74_dense_755_matmul_readvariableop_resource*
_output_shapes

:f*
dtype0È
sequential_74/dense_755/MatMulMatMul5sequential_74/leaky_re_lu_680/LeakyRelu:activations:05sequential_74/dense_755/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_74/dense_755/BiasAdd/ReadVariableOpReadVariableOp7sequential_74_dense_755_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_74/dense_755/BiasAddBiasAdd(sequential_74/dense_755/MatMul:product:06sequential_74/dense_755/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_74/dense_755/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp?^sequential_74/batch_normalization_676/batchnorm/ReadVariableOpA^sequential_74/batch_normalization_676/batchnorm/ReadVariableOp_1A^sequential_74/batch_normalization_676/batchnorm/ReadVariableOp_2C^sequential_74/batch_normalization_676/batchnorm/mul/ReadVariableOp?^sequential_74/batch_normalization_677/batchnorm/ReadVariableOpA^sequential_74/batch_normalization_677/batchnorm/ReadVariableOp_1A^sequential_74/batch_normalization_677/batchnorm/ReadVariableOp_2C^sequential_74/batch_normalization_677/batchnorm/mul/ReadVariableOp?^sequential_74/batch_normalization_678/batchnorm/ReadVariableOpA^sequential_74/batch_normalization_678/batchnorm/ReadVariableOp_1A^sequential_74/batch_normalization_678/batchnorm/ReadVariableOp_2C^sequential_74/batch_normalization_678/batchnorm/mul/ReadVariableOp?^sequential_74/batch_normalization_679/batchnorm/ReadVariableOpA^sequential_74/batch_normalization_679/batchnorm/ReadVariableOp_1A^sequential_74/batch_normalization_679/batchnorm/ReadVariableOp_2C^sequential_74/batch_normalization_679/batchnorm/mul/ReadVariableOp?^sequential_74/batch_normalization_680/batchnorm/ReadVariableOpA^sequential_74/batch_normalization_680/batchnorm/ReadVariableOp_1A^sequential_74/batch_normalization_680/batchnorm/ReadVariableOp_2C^sequential_74/batch_normalization_680/batchnorm/mul/ReadVariableOp/^sequential_74/dense_750/BiasAdd/ReadVariableOp.^sequential_74/dense_750/MatMul/ReadVariableOp/^sequential_74/dense_751/BiasAdd/ReadVariableOp.^sequential_74/dense_751/MatMul/ReadVariableOp/^sequential_74/dense_752/BiasAdd/ReadVariableOp.^sequential_74/dense_752/MatMul/ReadVariableOp/^sequential_74/dense_753/BiasAdd/ReadVariableOp.^sequential_74/dense_753/MatMul/ReadVariableOp/^sequential_74/dense_754/BiasAdd/ReadVariableOp.^sequential_74/dense_754/MatMul/ReadVariableOp/^sequential_74/dense_755/BiasAdd/ReadVariableOp.^sequential_74/dense_755/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
>sequential_74/batch_normalization_676/batchnorm/ReadVariableOp>sequential_74/batch_normalization_676/batchnorm/ReadVariableOp2
@sequential_74/batch_normalization_676/batchnorm/ReadVariableOp_1@sequential_74/batch_normalization_676/batchnorm/ReadVariableOp_12
@sequential_74/batch_normalization_676/batchnorm/ReadVariableOp_2@sequential_74/batch_normalization_676/batchnorm/ReadVariableOp_22
Bsequential_74/batch_normalization_676/batchnorm/mul/ReadVariableOpBsequential_74/batch_normalization_676/batchnorm/mul/ReadVariableOp2
>sequential_74/batch_normalization_677/batchnorm/ReadVariableOp>sequential_74/batch_normalization_677/batchnorm/ReadVariableOp2
@sequential_74/batch_normalization_677/batchnorm/ReadVariableOp_1@sequential_74/batch_normalization_677/batchnorm/ReadVariableOp_12
@sequential_74/batch_normalization_677/batchnorm/ReadVariableOp_2@sequential_74/batch_normalization_677/batchnorm/ReadVariableOp_22
Bsequential_74/batch_normalization_677/batchnorm/mul/ReadVariableOpBsequential_74/batch_normalization_677/batchnorm/mul/ReadVariableOp2
>sequential_74/batch_normalization_678/batchnorm/ReadVariableOp>sequential_74/batch_normalization_678/batchnorm/ReadVariableOp2
@sequential_74/batch_normalization_678/batchnorm/ReadVariableOp_1@sequential_74/batch_normalization_678/batchnorm/ReadVariableOp_12
@sequential_74/batch_normalization_678/batchnorm/ReadVariableOp_2@sequential_74/batch_normalization_678/batchnorm/ReadVariableOp_22
Bsequential_74/batch_normalization_678/batchnorm/mul/ReadVariableOpBsequential_74/batch_normalization_678/batchnorm/mul/ReadVariableOp2
>sequential_74/batch_normalization_679/batchnorm/ReadVariableOp>sequential_74/batch_normalization_679/batchnorm/ReadVariableOp2
@sequential_74/batch_normalization_679/batchnorm/ReadVariableOp_1@sequential_74/batch_normalization_679/batchnorm/ReadVariableOp_12
@sequential_74/batch_normalization_679/batchnorm/ReadVariableOp_2@sequential_74/batch_normalization_679/batchnorm/ReadVariableOp_22
Bsequential_74/batch_normalization_679/batchnorm/mul/ReadVariableOpBsequential_74/batch_normalization_679/batchnorm/mul/ReadVariableOp2
>sequential_74/batch_normalization_680/batchnorm/ReadVariableOp>sequential_74/batch_normalization_680/batchnorm/ReadVariableOp2
@sequential_74/batch_normalization_680/batchnorm/ReadVariableOp_1@sequential_74/batch_normalization_680/batchnorm/ReadVariableOp_12
@sequential_74/batch_normalization_680/batchnorm/ReadVariableOp_2@sequential_74/batch_normalization_680/batchnorm/ReadVariableOp_22
Bsequential_74/batch_normalization_680/batchnorm/mul/ReadVariableOpBsequential_74/batch_normalization_680/batchnorm/mul/ReadVariableOp2`
.sequential_74/dense_750/BiasAdd/ReadVariableOp.sequential_74/dense_750/BiasAdd/ReadVariableOp2^
-sequential_74/dense_750/MatMul/ReadVariableOp-sequential_74/dense_750/MatMul/ReadVariableOp2`
.sequential_74/dense_751/BiasAdd/ReadVariableOp.sequential_74/dense_751/BiasAdd/ReadVariableOp2^
-sequential_74/dense_751/MatMul/ReadVariableOp-sequential_74/dense_751/MatMul/ReadVariableOp2`
.sequential_74/dense_752/BiasAdd/ReadVariableOp.sequential_74/dense_752/BiasAdd/ReadVariableOp2^
-sequential_74/dense_752/MatMul/ReadVariableOp-sequential_74/dense_752/MatMul/ReadVariableOp2`
.sequential_74/dense_753/BiasAdd/ReadVariableOp.sequential_74/dense_753/BiasAdd/ReadVariableOp2^
-sequential_74/dense_753/MatMul/ReadVariableOp-sequential_74/dense_753/MatMul/ReadVariableOp2`
.sequential_74/dense_754/BiasAdd/ReadVariableOp.sequential_74/dense_754/BiasAdd/ReadVariableOp2^
-sequential_74/dense_754/MatMul/ReadVariableOp-sequential_74/dense_754/MatMul/ReadVariableOp2`
.sequential_74/dense_755/BiasAdd/ReadVariableOp.sequential_74/dense_755/BiasAdd/ReadVariableOp2^
-sequential_74/dense_755/MatMul/ReadVariableOp-sequential_74/dense_755/MatMul/ReadVariableOp:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_74_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ä

*__inference_dense_752_layer_call_fn_751656

inputs
unknown:.G
	unknown_0:G
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_752_layer_call_and_return_conditional_losses_750067o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
Ä

*__inference_dense_753_layer_call_fn_751765

inputs
unknown:GG
	unknown_0:G
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_753_layer_call_and_return_conditional_losses_750099o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
È	
ö
E__inference_dense_751_layer_call_and_return_conditional_losses_751557

inputs0
matmul_readvariableop_resource:..-
biasadd_readvariableop_resource:.
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:..*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:.*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ._
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_677_layer_call_fn_751642

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
:ÿÿÿÿÿÿÿÿÿ.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_677_layer_call_and_return_conditional_losses_750055`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ."
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_676_layer_call_and_return_conditional_losses_749640

inputs5
'assignmovingavg_readvariableop_resource:.7
)assignmovingavg_1_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:./
!batchnorm_readvariableop_resource:.
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
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

:.
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
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
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:.*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:.x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:.¬
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
:.*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:.~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:.´
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
:ÿÿÿÿÿÿÿÿÿ.h
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
:ÿÿÿÿÿÿÿÿÿ.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_680_layer_call_and_return_conditional_losses_751964

inputs5
'assignmovingavg_readvariableop_resource:f7
)assignmovingavg_1_readvariableop_resource:f3
%batchnorm_mul_readvariableop_resource:f/
!batchnorm_readvariableop_resource:f
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:f
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:f*
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
:f*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:fx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f¬
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
:f*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:f~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f´
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
:fP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:f~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:fc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:fv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:fr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿf: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_676_layer_call_and_return_conditional_losses_751494

inputs/
!batchnorm_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:.1
#batchnorm_readvariableop_1_resource:.1
#batchnorm_readvariableop_2_resource:.
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:.*
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
:ÿÿÿÿÿÿÿÿÿ.z
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
:ÿÿÿÿÿÿÿÿÿ.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
Ç§
Â'
__inference__traced_save_752273
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	/
+savev2_dense_750_kernel_read_readvariableop-
)savev2_dense_750_bias_read_readvariableop<
8savev2_batch_normalization_676_gamma_read_readvariableop;
7savev2_batch_normalization_676_beta_read_readvariableopB
>savev2_batch_normalization_676_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_676_moving_variance_read_readvariableop/
+savev2_dense_751_kernel_read_readvariableop-
)savev2_dense_751_bias_read_readvariableop<
8savev2_batch_normalization_677_gamma_read_readvariableop;
7savev2_batch_normalization_677_beta_read_readvariableopB
>savev2_batch_normalization_677_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_677_moving_variance_read_readvariableop/
+savev2_dense_752_kernel_read_readvariableop-
)savev2_dense_752_bias_read_readvariableop<
8savev2_batch_normalization_678_gamma_read_readvariableop;
7savev2_batch_normalization_678_beta_read_readvariableopB
>savev2_batch_normalization_678_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_678_moving_variance_read_readvariableop/
+savev2_dense_753_kernel_read_readvariableop-
)savev2_dense_753_bias_read_readvariableop<
8savev2_batch_normalization_679_gamma_read_readvariableop;
7savev2_batch_normalization_679_beta_read_readvariableopB
>savev2_batch_normalization_679_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_679_moving_variance_read_readvariableop/
+savev2_dense_754_kernel_read_readvariableop-
)savev2_dense_754_bias_read_readvariableop<
8savev2_batch_normalization_680_gamma_read_readvariableop;
7savev2_batch_normalization_680_beta_read_readvariableopB
>savev2_batch_normalization_680_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_680_moving_variance_read_readvariableop/
+savev2_dense_755_kernel_read_readvariableop-
)savev2_dense_755_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_750_kernel_m_read_readvariableop4
0savev2_adam_dense_750_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_676_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_676_beta_m_read_readvariableop6
2savev2_adam_dense_751_kernel_m_read_readvariableop4
0savev2_adam_dense_751_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_677_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_677_beta_m_read_readvariableop6
2savev2_adam_dense_752_kernel_m_read_readvariableop4
0savev2_adam_dense_752_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_678_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_678_beta_m_read_readvariableop6
2savev2_adam_dense_753_kernel_m_read_readvariableop4
0savev2_adam_dense_753_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_679_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_679_beta_m_read_readvariableop6
2savev2_adam_dense_754_kernel_m_read_readvariableop4
0savev2_adam_dense_754_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_680_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_680_beta_m_read_readvariableop6
2savev2_adam_dense_755_kernel_m_read_readvariableop4
0savev2_adam_dense_755_bias_m_read_readvariableop6
2savev2_adam_dense_750_kernel_v_read_readvariableop4
0savev2_adam_dense_750_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_676_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_676_beta_v_read_readvariableop6
2savev2_adam_dense_751_kernel_v_read_readvariableop4
0savev2_adam_dense_751_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_677_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_677_beta_v_read_readvariableop6
2savev2_adam_dense_752_kernel_v_read_readvariableop4
0savev2_adam_dense_752_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_678_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_678_beta_v_read_readvariableop6
2savev2_adam_dense_753_kernel_v_read_readvariableop4
0savev2_adam_dense_753_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_679_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_679_beta_v_read_readvariableop6
2savev2_adam_dense_754_kernel_v_read_readvariableop4
0savev2_adam_dense_754_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_680_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_680_beta_v_read_readvariableop6
2savev2_adam_dense_755_kernel_v_read_readvariableop4
0savev2_adam_dense_755_bias_v_read_readvariableop
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
: Á/
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*ê.
valueà.BÝ.VB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*Á
value·B´VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B &
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop+savev2_dense_750_kernel_read_readvariableop)savev2_dense_750_bias_read_readvariableop8savev2_batch_normalization_676_gamma_read_readvariableop7savev2_batch_normalization_676_beta_read_readvariableop>savev2_batch_normalization_676_moving_mean_read_readvariableopBsavev2_batch_normalization_676_moving_variance_read_readvariableop+savev2_dense_751_kernel_read_readvariableop)savev2_dense_751_bias_read_readvariableop8savev2_batch_normalization_677_gamma_read_readvariableop7savev2_batch_normalization_677_beta_read_readvariableop>savev2_batch_normalization_677_moving_mean_read_readvariableopBsavev2_batch_normalization_677_moving_variance_read_readvariableop+savev2_dense_752_kernel_read_readvariableop)savev2_dense_752_bias_read_readvariableop8savev2_batch_normalization_678_gamma_read_readvariableop7savev2_batch_normalization_678_beta_read_readvariableop>savev2_batch_normalization_678_moving_mean_read_readvariableopBsavev2_batch_normalization_678_moving_variance_read_readvariableop+savev2_dense_753_kernel_read_readvariableop)savev2_dense_753_bias_read_readvariableop8savev2_batch_normalization_679_gamma_read_readvariableop7savev2_batch_normalization_679_beta_read_readvariableop>savev2_batch_normalization_679_moving_mean_read_readvariableopBsavev2_batch_normalization_679_moving_variance_read_readvariableop+savev2_dense_754_kernel_read_readvariableop)savev2_dense_754_bias_read_readvariableop8savev2_batch_normalization_680_gamma_read_readvariableop7savev2_batch_normalization_680_beta_read_readvariableop>savev2_batch_normalization_680_moving_mean_read_readvariableopBsavev2_batch_normalization_680_moving_variance_read_readvariableop+savev2_dense_755_kernel_read_readvariableop)savev2_dense_755_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_750_kernel_m_read_readvariableop0savev2_adam_dense_750_bias_m_read_readvariableop?savev2_adam_batch_normalization_676_gamma_m_read_readvariableop>savev2_adam_batch_normalization_676_beta_m_read_readvariableop2savev2_adam_dense_751_kernel_m_read_readvariableop0savev2_adam_dense_751_bias_m_read_readvariableop?savev2_adam_batch_normalization_677_gamma_m_read_readvariableop>savev2_adam_batch_normalization_677_beta_m_read_readvariableop2savev2_adam_dense_752_kernel_m_read_readvariableop0savev2_adam_dense_752_bias_m_read_readvariableop?savev2_adam_batch_normalization_678_gamma_m_read_readvariableop>savev2_adam_batch_normalization_678_beta_m_read_readvariableop2savev2_adam_dense_753_kernel_m_read_readvariableop0savev2_adam_dense_753_bias_m_read_readvariableop?savev2_adam_batch_normalization_679_gamma_m_read_readvariableop>savev2_adam_batch_normalization_679_beta_m_read_readvariableop2savev2_adam_dense_754_kernel_m_read_readvariableop0savev2_adam_dense_754_bias_m_read_readvariableop?savev2_adam_batch_normalization_680_gamma_m_read_readvariableop>savev2_adam_batch_normalization_680_beta_m_read_readvariableop2savev2_adam_dense_755_kernel_m_read_readvariableop0savev2_adam_dense_755_bias_m_read_readvariableop2savev2_adam_dense_750_kernel_v_read_readvariableop0savev2_adam_dense_750_bias_v_read_readvariableop?savev2_adam_batch_normalization_676_gamma_v_read_readvariableop>savev2_adam_batch_normalization_676_beta_v_read_readvariableop2savev2_adam_dense_751_kernel_v_read_readvariableop0savev2_adam_dense_751_bias_v_read_readvariableop?savev2_adam_batch_normalization_677_gamma_v_read_readvariableop>savev2_adam_batch_normalization_677_beta_v_read_readvariableop2savev2_adam_dense_752_kernel_v_read_readvariableop0savev2_adam_dense_752_bias_v_read_readvariableop?savev2_adam_batch_normalization_678_gamma_v_read_readvariableop>savev2_adam_batch_normalization_678_beta_v_read_readvariableop2savev2_adam_dense_753_kernel_v_read_readvariableop0savev2_adam_dense_753_bias_v_read_readvariableop?savev2_adam_batch_normalization_679_gamma_v_read_readvariableop>savev2_adam_batch_normalization_679_beta_v_read_readvariableop2savev2_adam_dense_754_kernel_v_read_readvariableop0savev2_adam_dense_754_bias_v_read_readvariableop?savev2_adam_batch_normalization_680_gamma_v_read_readvariableop>savev2_adam_batch_normalization_680_beta_v_read_readvariableop2savev2_adam_dense_755_kernel_v_read_readvariableop0savev2_adam_dense_755_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *d
dtypesZ
X2V		
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

identity_1Identity_1:output:0*Ã
_input_shapes±
®: ::: :.:.:.:.:.:.:..:.:.:.:.:.:.G:G:G:G:G:G:GG:G:G:G:G:G:Gf:f:f:f:f:f:f:: : : : : : :.:.:.:.:..:.:.:.:.G:G:G:G:GG:G:G:G:Gf:f:f:f:f::.:.:.:.:..:.:.:.:.G:G:G:G:GG:G:G:G:Gf:f:f:f:f:: 2(
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

:.: 

_output_shapes
:.: 

_output_shapes
:.: 

_output_shapes
:.: 

_output_shapes
:.: 	

_output_shapes
:.:$
 

_output_shapes

:..: 

_output_shapes
:.: 

_output_shapes
:.: 

_output_shapes
:.: 

_output_shapes
:.: 

_output_shapes
:.:$ 

_output_shapes

:.G: 

_output_shapes
:G: 

_output_shapes
:G: 

_output_shapes
:G: 

_output_shapes
:G: 

_output_shapes
:G:$ 

_output_shapes

:GG: 

_output_shapes
:G: 

_output_shapes
:G: 

_output_shapes
:G: 

_output_shapes
:G: 

_output_shapes
:G:$ 

_output_shapes

:Gf: 

_output_shapes
:f: 

_output_shapes
:f: 

_output_shapes
:f:  

_output_shapes
:f: !

_output_shapes
:f:$" 

_output_shapes

:f: #

_output_shapes
::$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :$* 

_output_shapes

:.: +

_output_shapes
:.: ,

_output_shapes
:.: -

_output_shapes
:.:$. 

_output_shapes

:..: /

_output_shapes
:.: 0

_output_shapes
:.: 1

_output_shapes
:.:$2 

_output_shapes

:.G: 3

_output_shapes
:G: 4

_output_shapes
:G: 5

_output_shapes
:G:$6 

_output_shapes

:GG: 7

_output_shapes
:G: 8

_output_shapes
:G: 9

_output_shapes
:G:$: 

_output_shapes

:Gf: ;

_output_shapes
:f: <

_output_shapes
:f: =

_output_shapes
:f:$> 

_output_shapes

:f: ?

_output_shapes
::$@ 

_output_shapes

:.: A
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

:..: E

_output_shapes
:.: F

_output_shapes
:.: G

_output_shapes
:.:$H 

_output_shapes

:.G: I

_output_shapes
:G: J

_output_shapes
:G: K

_output_shapes
:G:$L 

_output_shapes

:GG: M

_output_shapes
:G: N

_output_shapes
:G: O

_output_shapes
:G:$P 

_output_shapes

:Gf: Q

_output_shapes
:f: R

_output_shapes
:f: S

_output_shapes
:f:$T 

_output_shapes

:f: U

_output_shapes
::V

_output_shapes
: 
%
ì
S__inference_batch_normalization_678_layer_call_and_return_conditional_losses_749804

inputs5
'assignmovingavg_readvariableop_resource:G7
)assignmovingavg_1_readvariableop_resource:G3
%batchnorm_mul_readvariableop_resource:G/
!batchnorm_readvariableop_resource:G
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:G*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:G
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:G*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:G*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:G*
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
:G*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Gx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:G¬
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
:G*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:G~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:G´
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
:GP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:G~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:G*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Gc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Gv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:G*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Gr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_676_layer_call_and_return_conditional_losses_751538

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ."
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_678_layer_call_and_return_conditional_losses_750087

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿG:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_677_layer_call_and_return_conditional_losses_751647

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ."
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
öY
£
I__inference_sequential_74_layer_call_and_return_conditional_losses_750170

inputs
normalization_74_sub_y
normalization_74_sqrt_x"
dense_750_750004:.
dense_750_750006:.,
batch_normalization_676_750009:.,
batch_normalization_676_750011:.,
batch_normalization_676_750013:.,
batch_normalization_676_750015:."
dense_751_750036:..
dense_751_750038:.,
batch_normalization_677_750041:.,
batch_normalization_677_750043:.,
batch_normalization_677_750045:.,
batch_normalization_677_750047:."
dense_752_750068:.G
dense_752_750070:G,
batch_normalization_678_750073:G,
batch_normalization_678_750075:G,
batch_normalization_678_750077:G,
batch_normalization_678_750079:G"
dense_753_750100:GG
dense_753_750102:G,
batch_normalization_679_750105:G,
batch_normalization_679_750107:G,
batch_normalization_679_750109:G,
batch_normalization_679_750111:G"
dense_754_750132:Gf
dense_754_750134:f,
batch_normalization_680_750137:f,
batch_normalization_680_750139:f,
batch_normalization_680_750141:f,
batch_normalization_680_750143:f"
dense_755_750164:f
dense_755_750166:
identity¢/batch_normalization_676/StatefulPartitionedCall¢/batch_normalization_677/StatefulPartitionedCall¢/batch_normalization_678/StatefulPartitionedCall¢/batch_normalization_679/StatefulPartitionedCall¢/batch_normalization_680/StatefulPartitionedCall¢!dense_750/StatefulPartitionedCall¢!dense_751/StatefulPartitionedCall¢!dense_752/StatefulPartitionedCall¢!dense_753/StatefulPartitionedCall¢!dense_754/StatefulPartitionedCall¢!dense_755/StatefulPartitionedCallm
normalization_74/subSubinputsnormalization_74_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_74/SqrtSqrtnormalization_74_sqrt_x*
T0*
_output_shapes

:_
normalization_74/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_74/MaximumMaximumnormalization_74/Sqrt:y:0#normalization_74/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_74/truedivRealDivnormalization_74/sub:z:0normalization_74/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_750/StatefulPartitionedCallStatefulPartitionedCallnormalization_74/truediv:z:0dense_750_750004dense_750_750006*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_750_layer_call_and_return_conditional_losses_750003
/batch_normalization_676/StatefulPartitionedCallStatefulPartitionedCall*dense_750/StatefulPartitionedCall:output:0batch_normalization_676_750009batch_normalization_676_750011batch_normalization_676_750013batch_normalization_676_750015*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_676_layer_call_and_return_conditional_losses_749593ø
leaky_re_lu_676/PartitionedCallPartitionedCall8batch_normalization_676/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_676_layer_call_and_return_conditional_losses_750023
!dense_751/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_676/PartitionedCall:output:0dense_751_750036dense_751_750038*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_751_layer_call_and_return_conditional_losses_750035
/batch_normalization_677/StatefulPartitionedCallStatefulPartitionedCall*dense_751/StatefulPartitionedCall:output:0batch_normalization_677_750041batch_normalization_677_750043batch_normalization_677_750045batch_normalization_677_750047*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_677_layer_call_and_return_conditional_losses_749675ø
leaky_re_lu_677/PartitionedCallPartitionedCall8batch_normalization_677/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_677_layer_call_and_return_conditional_losses_750055
!dense_752/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_677/PartitionedCall:output:0dense_752_750068dense_752_750070*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_752_layer_call_and_return_conditional_losses_750067
/batch_normalization_678/StatefulPartitionedCallStatefulPartitionedCall*dense_752/StatefulPartitionedCall:output:0batch_normalization_678_750073batch_normalization_678_750075batch_normalization_678_750077batch_normalization_678_750079*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_678_layer_call_and_return_conditional_losses_749757ø
leaky_re_lu_678/PartitionedCallPartitionedCall8batch_normalization_678/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_678_layer_call_and_return_conditional_losses_750087
!dense_753/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_678/PartitionedCall:output:0dense_753_750100dense_753_750102*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_753_layer_call_and_return_conditional_losses_750099
/batch_normalization_679/StatefulPartitionedCallStatefulPartitionedCall*dense_753/StatefulPartitionedCall:output:0batch_normalization_679_750105batch_normalization_679_750107batch_normalization_679_750109batch_normalization_679_750111*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_679_layer_call_and_return_conditional_losses_749839ø
leaky_re_lu_679/PartitionedCallPartitionedCall8batch_normalization_679/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_679_layer_call_and_return_conditional_losses_750119
!dense_754/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_679/PartitionedCall:output:0dense_754_750132dense_754_750134*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_754_layer_call_and_return_conditional_losses_750131
/batch_normalization_680/StatefulPartitionedCallStatefulPartitionedCall*dense_754/StatefulPartitionedCall:output:0batch_normalization_680_750137batch_normalization_680_750139batch_normalization_680_750141batch_normalization_680_750143*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_680_layer_call_and_return_conditional_losses_749921ø
leaky_re_lu_680/PartitionedCallPartitionedCall8batch_normalization_680/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_680_layer_call_and_return_conditional_losses_750151
!dense_755/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_680/PartitionedCall:output:0dense_755_750164dense_755_750166*
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
E__inference_dense_755_layer_call_and_return_conditional_losses_750163y
IdentityIdentity*dense_755/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_676/StatefulPartitionedCall0^batch_normalization_677/StatefulPartitionedCall0^batch_normalization_678/StatefulPartitionedCall0^batch_normalization_679/StatefulPartitionedCall0^batch_normalization_680/StatefulPartitionedCall"^dense_750/StatefulPartitionedCall"^dense_751/StatefulPartitionedCall"^dense_752/StatefulPartitionedCall"^dense_753/StatefulPartitionedCall"^dense_754/StatefulPartitionedCall"^dense_755/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_676/StatefulPartitionedCall/batch_normalization_676/StatefulPartitionedCall2b
/batch_normalization_677/StatefulPartitionedCall/batch_normalization_677/StatefulPartitionedCall2b
/batch_normalization_678/StatefulPartitionedCall/batch_normalization_678/StatefulPartitionedCall2b
/batch_normalization_679/StatefulPartitionedCall/batch_normalization_679/StatefulPartitionedCall2b
/batch_normalization_680/StatefulPartitionedCall/batch_normalization_680/StatefulPartitionedCall2F
!dense_750/StatefulPartitionedCall!dense_750/StatefulPartitionedCall2F
!dense_751/StatefulPartitionedCall!dense_751/StatefulPartitionedCall2F
!dense_752/StatefulPartitionedCall!dense_752/StatefulPartitionedCall2F
!dense_753/StatefulPartitionedCall!dense_753/StatefulPartitionedCall2F
!dense_754/StatefulPartitionedCall!dense_754/StatefulPartitionedCall2F
!dense_755/StatefulPartitionedCall!dense_755/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ð
²
S__inference_batch_normalization_679_layer_call_and_return_conditional_losses_751821

inputs/
!batchnorm_readvariableop_resource:G3
%batchnorm_mul_readvariableop_resource:G1
#batchnorm_readvariableop_1_resource:G1
#batchnorm_readvariableop_2_resource:G
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:G*
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
:GP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:G~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:G*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Gc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:G*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Gz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:G*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Gr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
Ä

*__inference_dense_754_layer_call_fn_751874

inputs
unknown:Gf
	unknown_0:f
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_754_layer_call_and_return_conditional_losses_750131o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_677_layer_call_fn_751570

inputs
unknown:.
	unknown_0:.
	unknown_1:.
	unknown_2:.
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_677_layer_call_and_return_conditional_losses_749675o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_677_layer_call_fn_751583

inputs
unknown:.
	unknown_0:.
	unknown_1:.
	unknown_2:.
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_677_layer_call_and_return_conditional_losses_749722o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_680_layer_call_and_return_conditional_losses_749921

inputs/
!batchnorm_readvariableop_resource:f3
%batchnorm_mul_readvariableop_resource:f1
#batchnorm_readvariableop_1_resource:f1
#batchnorm_readvariableop_2_resource:f
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:f*
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
:fP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:f~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:fc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:fz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:fr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿf: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_677_layer_call_and_return_conditional_losses_751603

inputs/
!batchnorm_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:.1
#batchnorm_readvariableop_1_resource:.1
#batchnorm_readvariableop_2_resource:.
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:.*
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
:ÿÿÿÿÿÿÿÿÿ.z
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
:ÿÿÿÿÿÿÿÿÿ.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_676_layer_call_fn_751533

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
:ÿÿÿÿÿÿÿÿÿ.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_676_layer_call_and_return_conditional_losses_750023`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ."
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
È	
ö
E__inference_dense_750_layer_call_and_return_conditional_losses_751448

inputs0
matmul_readvariableop_resource:.-
biasadd_readvariableop_resource:.
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:.*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:.*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ._
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.w
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
Ð
²
S__inference_batch_normalization_677_layer_call_and_return_conditional_losses_749675

inputs/
!batchnorm_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:.1
#batchnorm_readvariableop_1_resource:.1
#batchnorm_readvariableop_2_resource:.
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:.*
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
:ÿÿÿÿÿÿÿÿÿ.z
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
:ÿÿÿÿÿÿÿÿÿ.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_677_layer_call_and_return_conditional_losses_750055

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ."
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_680_layer_call_and_return_conditional_losses_751930

inputs/
!batchnorm_readvariableop_resource:f3
%batchnorm_mul_readvariableop_resource:f1
#batchnorm_readvariableop_1_resource:f1
#batchnorm_readvariableop_2_resource:f
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:f*
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
:fP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:f~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:fc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:fz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:fr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿf: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_679_layer_call_and_return_conditional_losses_749886

inputs5
'assignmovingavg_readvariableop_resource:G7
)assignmovingavg_1_readvariableop_resource:G3
%batchnorm_mul_readvariableop_resource:G/
!batchnorm_readvariableop_resource:G
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:G*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:G
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:G*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:G*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:G*
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
:G*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Gx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:G¬
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
:G*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:G~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:G´
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
:GP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:G~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:G*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Gc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Gv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:G*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Gr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
È	
ö
E__inference_dense_754_layer_call_and_return_conditional_losses_751884

inputs0
matmul_readvariableop_resource:Gf-
biasadd_readvariableop_resource:f
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Gf*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:f*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_676_layer_call_fn_751461

inputs
unknown:.
	unknown_0:.
	unknown_1:.
	unknown_2:.
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_676_layer_call_and_return_conditional_losses_749593o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_676_layer_call_and_return_conditional_losses_751528

inputs5
'assignmovingavg_readvariableop_resource:.7
)assignmovingavg_1_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:./
!batchnorm_readvariableop_resource:.
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
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

:.
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
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
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:.*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:.x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:.¬
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
:.*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:.~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:.´
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
:ÿÿÿÿÿÿÿÿÿ.h
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
:ÿÿÿÿÿÿÿÿÿ.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
È	
ö
E__inference_dense_752_layer_call_and_return_conditional_losses_751666

inputs0
matmul_readvariableop_resource:.G-
biasadd_readvariableop_resource:G
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:.G*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:G*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
Ô
Ò
.__inference_sequential_74_layer_call_fn_750900

inputs
unknown
	unknown_0
	unknown_1:.
	unknown_2:.
	unknown_3:.
	unknown_4:.
	unknown_5:.
	unknown_6:.
	unknown_7:..
	unknown_8:.
	unknown_9:.

unknown_10:.

unknown_11:.

unknown_12:.

unknown_13:.G

unknown_14:G

unknown_15:G

unknown_16:G

unknown_17:G

unknown_18:G

unknown_19:GG

unknown_20:G

unknown_21:G

unknown_22:G

unknown_23:G

unknown_24:G

unknown_25:Gf

unknown_26:f

unknown_27:f

unknown_28:f

unknown_29:f

unknown_30:f

unknown_31:f

unknown_32:
identity¢StatefulPartitionedCall
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
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 !"*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_74_layer_call_and_return_conditional_losses_750170o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
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
K__inference_leaky_re_lu_680_layer_call_and_return_conditional_losses_751974

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿf:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
¦Z
³
I__inference_sequential_74_layer_call_and_return_conditional_losses_750732
normalization_74_input
normalization_74_sub_y
normalization_74_sqrt_x"
dense_750_750651:.
dense_750_750653:.,
batch_normalization_676_750656:.,
batch_normalization_676_750658:.,
batch_normalization_676_750660:.,
batch_normalization_676_750662:."
dense_751_750666:..
dense_751_750668:.,
batch_normalization_677_750671:.,
batch_normalization_677_750673:.,
batch_normalization_677_750675:.,
batch_normalization_677_750677:."
dense_752_750681:.G
dense_752_750683:G,
batch_normalization_678_750686:G,
batch_normalization_678_750688:G,
batch_normalization_678_750690:G,
batch_normalization_678_750692:G"
dense_753_750696:GG
dense_753_750698:G,
batch_normalization_679_750701:G,
batch_normalization_679_750703:G,
batch_normalization_679_750705:G,
batch_normalization_679_750707:G"
dense_754_750711:Gf
dense_754_750713:f,
batch_normalization_680_750716:f,
batch_normalization_680_750718:f,
batch_normalization_680_750720:f,
batch_normalization_680_750722:f"
dense_755_750726:f
dense_755_750728:
identity¢/batch_normalization_676/StatefulPartitionedCall¢/batch_normalization_677/StatefulPartitionedCall¢/batch_normalization_678/StatefulPartitionedCall¢/batch_normalization_679/StatefulPartitionedCall¢/batch_normalization_680/StatefulPartitionedCall¢!dense_750/StatefulPartitionedCall¢!dense_751/StatefulPartitionedCall¢!dense_752/StatefulPartitionedCall¢!dense_753/StatefulPartitionedCall¢!dense_754/StatefulPartitionedCall¢!dense_755/StatefulPartitionedCall}
normalization_74/subSubnormalization_74_inputnormalization_74_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_74/SqrtSqrtnormalization_74_sqrt_x*
T0*
_output_shapes

:_
normalization_74/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_74/MaximumMaximumnormalization_74/Sqrt:y:0#normalization_74/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_74/truedivRealDivnormalization_74/sub:z:0normalization_74/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_750/StatefulPartitionedCallStatefulPartitionedCallnormalization_74/truediv:z:0dense_750_750651dense_750_750653*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_750_layer_call_and_return_conditional_losses_750003
/batch_normalization_676/StatefulPartitionedCallStatefulPartitionedCall*dense_750/StatefulPartitionedCall:output:0batch_normalization_676_750656batch_normalization_676_750658batch_normalization_676_750660batch_normalization_676_750662*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_676_layer_call_and_return_conditional_losses_749593ø
leaky_re_lu_676/PartitionedCallPartitionedCall8batch_normalization_676/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_676_layer_call_and_return_conditional_losses_750023
!dense_751/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_676/PartitionedCall:output:0dense_751_750666dense_751_750668*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_751_layer_call_and_return_conditional_losses_750035
/batch_normalization_677/StatefulPartitionedCallStatefulPartitionedCall*dense_751/StatefulPartitionedCall:output:0batch_normalization_677_750671batch_normalization_677_750673batch_normalization_677_750675batch_normalization_677_750677*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_677_layer_call_and_return_conditional_losses_749675ø
leaky_re_lu_677/PartitionedCallPartitionedCall8batch_normalization_677/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_677_layer_call_and_return_conditional_losses_750055
!dense_752/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_677/PartitionedCall:output:0dense_752_750681dense_752_750683*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_752_layer_call_and_return_conditional_losses_750067
/batch_normalization_678/StatefulPartitionedCallStatefulPartitionedCall*dense_752/StatefulPartitionedCall:output:0batch_normalization_678_750686batch_normalization_678_750688batch_normalization_678_750690batch_normalization_678_750692*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_678_layer_call_and_return_conditional_losses_749757ø
leaky_re_lu_678/PartitionedCallPartitionedCall8batch_normalization_678/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_678_layer_call_and_return_conditional_losses_750087
!dense_753/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_678/PartitionedCall:output:0dense_753_750696dense_753_750698*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_753_layer_call_and_return_conditional_losses_750099
/batch_normalization_679/StatefulPartitionedCallStatefulPartitionedCall*dense_753/StatefulPartitionedCall:output:0batch_normalization_679_750701batch_normalization_679_750703batch_normalization_679_750705batch_normalization_679_750707*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_679_layer_call_and_return_conditional_losses_749839ø
leaky_re_lu_679/PartitionedCallPartitionedCall8batch_normalization_679/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_679_layer_call_and_return_conditional_losses_750119
!dense_754/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_679/PartitionedCall:output:0dense_754_750711dense_754_750713*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_754_layer_call_and_return_conditional_losses_750131
/batch_normalization_680/StatefulPartitionedCallStatefulPartitionedCall*dense_754/StatefulPartitionedCall:output:0batch_normalization_680_750716batch_normalization_680_750718batch_normalization_680_750720batch_normalization_680_750722*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_680_layer_call_and_return_conditional_losses_749921ø
leaky_re_lu_680/PartitionedCallPartitionedCall8batch_normalization_680/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_680_layer_call_and_return_conditional_losses_750151
!dense_755/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_680/PartitionedCall:output:0dense_755_750726dense_755_750728*
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
E__inference_dense_755_layer_call_and_return_conditional_losses_750163y
IdentityIdentity*dense_755/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_676/StatefulPartitionedCall0^batch_normalization_677/StatefulPartitionedCall0^batch_normalization_678/StatefulPartitionedCall0^batch_normalization_679/StatefulPartitionedCall0^batch_normalization_680/StatefulPartitionedCall"^dense_750/StatefulPartitionedCall"^dense_751/StatefulPartitionedCall"^dense_752/StatefulPartitionedCall"^dense_753/StatefulPartitionedCall"^dense_754/StatefulPartitionedCall"^dense_755/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_676/StatefulPartitionedCall/batch_normalization_676/StatefulPartitionedCall2b
/batch_normalization_677/StatefulPartitionedCall/batch_normalization_677/StatefulPartitionedCall2b
/batch_normalization_678/StatefulPartitionedCall/batch_normalization_678/StatefulPartitionedCall2b
/batch_normalization_679/StatefulPartitionedCall/batch_normalization_679/StatefulPartitionedCall2b
/batch_normalization_680/StatefulPartitionedCall/batch_normalization_680/StatefulPartitionedCall2F
!dense_750/StatefulPartitionedCall!dense_750/StatefulPartitionedCall2F
!dense_751/StatefulPartitionedCall!dense_751/StatefulPartitionedCall2F
!dense_752/StatefulPartitionedCall!dense_752/StatefulPartitionedCall2F
!dense_753/StatefulPartitionedCall!dense_753/StatefulPartitionedCall2F
!dense_754/StatefulPartitionedCall!dense_754/StatefulPartitionedCall2F
!dense_755/StatefulPartitionedCall!dense_755/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_74_input:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_679_layer_call_and_return_conditional_losses_750119

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿG:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_678_layer_call_and_return_conditional_losses_751712

inputs/
!batchnorm_readvariableop_resource:G3
%batchnorm_mul_readvariableop_resource:G1
#batchnorm_readvariableop_1_resource:G1
#batchnorm_readvariableop_2_resource:G
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:G*
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
:GP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:G~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:G*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Gc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:G*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Gz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:G*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Gr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_679_layer_call_and_return_conditional_losses_751855

inputs5
'assignmovingavg_readvariableop_resource:G7
)assignmovingavg_1_readvariableop_resource:G3
%batchnorm_mul_readvariableop_resource:G/
!batchnorm_readvariableop_resource:G
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:G*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:G
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:G*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:G*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:G*
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
:G*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Gx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:G¬
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
:G*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:G~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:G´
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
:GP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:G~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:G*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Gc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Gv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:G*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Gr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_680_layer_call_fn_751910

inputs
unknown:f
	unknown_0:f
	unknown_1:f
	unknown_2:f
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_680_layer_call_and_return_conditional_losses_749968o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿf: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_677_layer_call_and_return_conditional_losses_751637

inputs5
'assignmovingavg_readvariableop_resource:.7
)assignmovingavg_1_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:./
!batchnorm_readvariableop_resource:.
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
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

:.
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
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
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:.*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:.x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:.¬
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
:.*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:.~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:.´
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
:ÿÿÿÿÿÿÿÿÿ.h
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
:ÿÿÿÿÿÿÿÿÿ.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_676_layer_call_fn_751474

inputs
unknown:.
	unknown_0:.
	unknown_1:.
	unknown_2:.
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_676_layer_call_and_return_conditional_losses_749640o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_680_layer_call_and_return_conditional_losses_750151

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿf:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
È	
ö
E__inference_dense_753_layer_call_and_return_conditional_losses_750099

inputs0
matmul_readvariableop_resource:GG-
biasadd_readvariableop_resource:G
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:GG*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:G*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_676_layer_call_and_return_conditional_losses_750023

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ."
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_679_layer_call_and_return_conditional_losses_751865

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿG:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_679_layer_call_and_return_conditional_losses_749839

inputs/
!batchnorm_readvariableop_resource:G3
%batchnorm_mul_readvariableop_resource:G1
#batchnorm_readvariableop_1_resource:G1
#batchnorm_readvariableop_2_resource:G
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:G*
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
:GP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:G~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:G*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Gc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:G*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Gz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:G*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Gr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_680_layer_call_fn_751897

inputs
unknown:f
	unknown_0:f
	unknown_1:f
	unknown_2:f
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_680_layer_call_and_return_conditional_losses_749921o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿf: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
Ò
Ø
$__inference_signature_wrapper_751382
normalization_74_input
unknown
	unknown_0
	unknown_1:.
	unknown_2:.
	unknown_3:.
	unknown_4:.
	unknown_5:.
	unknown_6:.
	unknown_7:..
	unknown_8:.
	unknown_9:.

unknown_10:.

unknown_11:.

unknown_12:.

unknown_13:.G

unknown_14:G

unknown_15:G

unknown_16:G

unknown_17:G

unknown_18:G

unknown_19:GG

unknown_20:G

unknown_21:G

unknown_22:G

unknown_23:G

unknown_24:G

unknown_25:Gf

unknown_26:f

unknown_27:f

unknown_28:f

unknown_29:f

unknown_30:f

unknown_31:f

unknown_32:
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallnormalization_74_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 !"*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_749569o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_74_input:$ 

_output_shapes

::$ 

_output_shapes

:
ª
Ó
8__inference_batch_normalization_678_layer_call_fn_751692

inputs
unknown:G
	unknown_0:G
	unknown_1:G
	unknown_2:G
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_678_layer_call_and_return_conditional_losses_749804o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
ú
â
.__inference_sequential_74_layer_call_fn_750641
normalization_74_input
unknown
	unknown_0
	unknown_1:.
	unknown_2:.
	unknown_3:.
	unknown_4:.
	unknown_5:.
	unknown_6:.
	unknown_7:..
	unknown_8:.
	unknown_9:.

unknown_10:.

unknown_11:.

unknown_12:.

unknown_13:.G

unknown_14:G

unknown_15:G

unknown_16:G

unknown_17:G

unknown_18:G

unknown_19:GG

unknown_20:G

unknown_21:G

unknown_22:G

unknown_23:G

unknown_24:G

unknown_25:Gf

unknown_26:f

unknown_27:f

unknown_28:f

unknown_29:f

unknown_30:f

unknown_31:f

unknown_32:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallnormalization_74_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
 !"*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_74_layer_call_and_return_conditional_losses_750497o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_74_input:$ 

_output_shapes

::$ 

_output_shapes

:
×º
ý
I__inference_sequential_74_layer_call_and_return_conditional_losses_751105

inputs
normalization_74_sub_y
normalization_74_sqrt_x:
(dense_750_matmul_readvariableop_resource:.7
)dense_750_biasadd_readvariableop_resource:.G
9batch_normalization_676_batchnorm_readvariableop_resource:.K
=batch_normalization_676_batchnorm_mul_readvariableop_resource:.I
;batch_normalization_676_batchnorm_readvariableop_1_resource:.I
;batch_normalization_676_batchnorm_readvariableop_2_resource:.:
(dense_751_matmul_readvariableop_resource:..7
)dense_751_biasadd_readvariableop_resource:.G
9batch_normalization_677_batchnorm_readvariableop_resource:.K
=batch_normalization_677_batchnorm_mul_readvariableop_resource:.I
;batch_normalization_677_batchnorm_readvariableop_1_resource:.I
;batch_normalization_677_batchnorm_readvariableop_2_resource:.:
(dense_752_matmul_readvariableop_resource:.G7
)dense_752_biasadd_readvariableop_resource:GG
9batch_normalization_678_batchnorm_readvariableop_resource:GK
=batch_normalization_678_batchnorm_mul_readvariableop_resource:GI
;batch_normalization_678_batchnorm_readvariableop_1_resource:GI
;batch_normalization_678_batchnorm_readvariableop_2_resource:G:
(dense_753_matmul_readvariableop_resource:GG7
)dense_753_biasadd_readvariableop_resource:GG
9batch_normalization_679_batchnorm_readvariableop_resource:GK
=batch_normalization_679_batchnorm_mul_readvariableop_resource:GI
;batch_normalization_679_batchnorm_readvariableop_1_resource:GI
;batch_normalization_679_batchnorm_readvariableop_2_resource:G:
(dense_754_matmul_readvariableop_resource:Gf7
)dense_754_biasadd_readvariableop_resource:fG
9batch_normalization_680_batchnorm_readvariableop_resource:fK
=batch_normalization_680_batchnorm_mul_readvariableop_resource:fI
;batch_normalization_680_batchnorm_readvariableop_1_resource:fI
;batch_normalization_680_batchnorm_readvariableop_2_resource:f:
(dense_755_matmul_readvariableop_resource:f7
)dense_755_biasadd_readvariableop_resource:
identity¢0batch_normalization_676/batchnorm/ReadVariableOp¢2batch_normalization_676/batchnorm/ReadVariableOp_1¢2batch_normalization_676/batchnorm/ReadVariableOp_2¢4batch_normalization_676/batchnorm/mul/ReadVariableOp¢0batch_normalization_677/batchnorm/ReadVariableOp¢2batch_normalization_677/batchnorm/ReadVariableOp_1¢2batch_normalization_677/batchnorm/ReadVariableOp_2¢4batch_normalization_677/batchnorm/mul/ReadVariableOp¢0batch_normalization_678/batchnorm/ReadVariableOp¢2batch_normalization_678/batchnorm/ReadVariableOp_1¢2batch_normalization_678/batchnorm/ReadVariableOp_2¢4batch_normalization_678/batchnorm/mul/ReadVariableOp¢0batch_normalization_679/batchnorm/ReadVariableOp¢2batch_normalization_679/batchnorm/ReadVariableOp_1¢2batch_normalization_679/batchnorm/ReadVariableOp_2¢4batch_normalization_679/batchnorm/mul/ReadVariableOp¢0batch_normalization_680/batchnorm/ReadVariableOp¢2batch_normalization_680/batchnorm/ReadVariableOp_1¢2batch_normalization_680/batchnorm/ReadVariableOp_2¢4batch_normalization_680/batchnorm/mul/ReadVariableOp¢ dense_750/BiasAdd/ReadVariableOp¢dense_750/MatMul/ReadVariableOp¢ dense_751/BiasAdd/ReadVariableOp¢dense_751/MatMul/ReadVariableOp¢ dense_752/BiasAdd/ReadVariableOp¢dense_752/MatMul/ReadVariableOp¢ dense_753/BiasAdd/ReadVariableOp¢dense_753/MatMul/ReadVariableOp¢ dense_754/BiasAdd/ReadVariableOp¢dense_754/MatMul/ReadVariableOp¢ dense_755/BiasAdd/ReadVariableOp¢dense_755/MatMul/ReadVariableOpm
normalization_74/subSubinputsnormalization_74_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_74/SqrtSqrtnormalization_74_sqrt_x*
T0*
_output_shapes

:_
normalization_74/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_74/MaximumMaximumnormalization_74/Sqrt:y:0#normalization_74/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_74/truedivRealDivnormalization_74/sub:z:0normalization_74/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_750/MatMul/ReadVariableOpReadVariableOp(dense_750_matmul_readvariableop_resource*
_output_shapes

:.*
dtype0
dense_750/MatMulMatMulnormalization_74/truediv:z:0'dense_750/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 dense_750/BiasAdd/ReadVariableOpReadVariableOp)dense_750_biasadd_readvariableop_resource*
_output_shapes
:.*
dtype0
dense_750/BiasAddBiasAdddense_750/MatMul:product:0(dense_750/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.¦
0batch_normalization_676/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_676_batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0l
'batch_normalization_676/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_676/batchnorm/addAddV28batch_normalization_676/batchnorm/ReadVariableOp:value:00batch_normalization_676/batchnorm/add/y:output:0*
T0*
_output_shapes
:.
'batch_normalization_676/batchnorm/RsqrtRsqrt)batch_normalization_676/batchnorm/add:z:0*
T0*
_output_shapes
:.®
4batch_normalization_676/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_676_batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0¼
%batch_normalization_676/batchnorm/mulMul+batch_normalization_676/batchnorm/Rsqrt:y:0<batch_normalization_676/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.§
'batch_normalization_676/batchnorm/mul_1Muldense_750/BiasAdd:output:0)batch_normalization_676/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.ª
2batch_normalization_676/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_676_batchnorm_readvariableop_1_resource*
_output_shapes
:.*
dtype0º
'batch_normalization_676/batchnorm/mul_2Mul:batch_normalization_676/batchnorm/ReadVariableOp_1:value:0)batch_normalization_676/batchnorm/mul:z:0*
T0*
_output_shapes
:.ª
2batch_normalization_676/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_676_batchnorm_readvariableop_2_resource*
_output_shapes
:.*
dtype0º
%batch_normalization_676/batchnorm/subSub:batch_normalization_676/batchnorm/ReadVariableOp_2:value:0+batch_normalization_676/batchnorm/mul_2:z:0*
T0*
_output_shapes
:.º
'batch_normalization_676/batchnorm/add_1AddV2+batch_normalization_676/batchnorm/mul_1:z:0)batch_normalization_676/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
leaky_re_lu_676/LeakyRelu	LeakyRelu+batch_normalization_676/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
alpha%>
dense_751/MatMul/ReadVariableOpReadVariableOp(dense_751_matmul_readvariableop_resource*
_output_shapes

:..*
dtype0
dense_751/MatMulMatMul'leaky_re_lu_676/LeakyRelu:activations:0'dense_751/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 dense_751/BiasAdd/ReadVariableOpReadVariableOp)dense_751_biasadd_readvariableop_resource*
_output_shapes
:.*
dtype0
dense_751/BiasAddBiasAdddense_751/MatMul:product:0(dense_751/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.¦
0batch_normalization_677/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_677_batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0l
'batch_normalization_677/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_677/batchnorm/addAddV28batch_normalization_677/batchnorm/ReadVariableOp:value:00batch_normalization_677/batchnorm/add/y:output:0*
T0*
_output_shapes
:.
'batch_normalization_677/batchnorm/RsqrtRsqrt)batch_normalization_677/batchnorm/add:z:0*
T0*
_output_shapes
:.®
4batch_normalization_677/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_677_batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0¼
%batch_normalization_677/batchnorm/mulMul+batch_normalization_677/batchnorm/Rsqrt:y:0<batch_normalization_677/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.§
'batch_normalization_677/batchnorm/mul_1Muldense_751/BiasAdd:output:0)batch_normalization_677/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.ª
2batch_normalization_677/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_677_batchnorm_readvariableop_1_resource*
_output_shapes
:.*
dtype0º
'batch_normalization_677/batchnorm/mul_2Mul:batch_normalization_677/batchnorm/ReadVariableOp_1:value:0)batch_normalization_677/batchnorm/mul:z:0*
T0*
_output_shapes
:.ª
2batch_normalization_677/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_677_batchnorm_readvariableop_2_resource*
_output_shapes
:.*
dtype0º
%batch_normalization_677/batchnorm/subSub:batch_normalization_677/batchnorm/ReadVariableOp_2:value:0+batch_normalization_677/batchnorm/mul_2:z:0*
T0*
_output_shapes
:.º
'batch_normalization_677/batchnorm/add_1AddV2+batch_normalization_677/batchnorm/mul_1:z:0)batch_normalization_677/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
leaky_re_lu_677/LeakyRelu	LeakyRelu+batch_normalization_677/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
alpha%>
dense_752/MatMul/ReadVariableOpReadVariableOp(dense_752_matmul_readvariableop_resource*
_output_shapes

:.G*
dtype0
dense_752/MatMulMatMul'leaky_re_lu_677/LeakyRelu:activations:0'dense_752/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 dense_752/BiasAdd/ReadVariableOpReadVariableOp)dense_752_biasadd_readvariableop_resource*
_output_shapes
:G*
dtype0
dense_752/BiasAddBiasAdddense_752/MatMul:product:0(dense_752/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG¦
0batch_normalization_678/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_678_batchnorm_readvariableop_resource*
_output_shapes
:G*
dtype0l
'batch_normalization_678/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_678/batchnorm/addAddV28batch_normalization_678/batchnorm/ReadVariableOp:value:00batch_normalization_678/batchnorm/add/y:output:0*
T0*
_output_shapes
:G
'batch_normalization_678/batchnorm/RsqrtRsqrt)batch_normalization_678/batchnorm/add:z:0*
T0*
_output_shapes
:G®
4batch_normalization_678/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_678_batchnorm_mul_readvariableop_resource*
_output_shapes
:G*
dtype0¼
%batch_normalization_678/batchnorm/mulMul+batch_normalization_678/batchnorm/Rsqrt:y:0<batch_normalization_678/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:G§
'batch_normalization_678/batchnorm/mul_1Muldense_752/BiasAdd:output:0)batch_normalization_678/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGª
2batch_normalization_678/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_678_batchnorm_readvariableop_1_resource*
_output_shapes
:G*
dtype0º
'batch_normalization_678/batchnorm/mul_2Mul:batch_normalization_678/batchnorm/ReadVariableOp_1:value:0)batch_normalization_678/batchnorm/mul:z:0*
T0*
_output_shapes
:Gª
2batch_normalization_678/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_678_batchnorm_readvariableop_2_resource*
_output_shapes
:G*
dtype0º
%batch_normalization_678/batchnorm/subSub:batch_normalization_678/batchnorm/ReadVariableOp_2:value:0+batch_normalization_678/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Gº
'batch_normalization_678/batchnorm/add_1AddV2+batch_normalization_678/batchnorm/mul_1:z:0)batch_normalization_678/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
leaky_re_lu_678/LeakyRelu	LeakyRelu+batch_normalization_678/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
alpha%>
dense_753/MatMul/ReadVariableOpReadVariableOp(dense_753_matmul_readvariableop_resource*
_output_shapes

:GG*
dtype0
dense_753/MatMulMatMul'leaky_re_lu_678/LeakyRelu:activations:0'dense_753/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 dense_753/BiasAdd/ReadVariableOpReadVariableOp)dense_753_biasadd_readvariableop_resource*
_output_shapes
:G*
dtype0
dense_753/BiasAddBiasAdddense_753/MatMul:product:0(dense_753/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG¦
0batch_normalization_679/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_679_batchnorm_readvariableop_resource*
_output_shapes
:G*
dtype0l
'batch_normalization_679/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_679/batchnorm/addAddV28batch_normalization_679/batchnorm/ReadVariableOp:value:00batch_normalization_679/batchnorm/add/y:output:0*
T0*
_output_shapes
:G
'batch_normalization_679/batchnorm/RsqrtRsqrt)batch_normalization_679/batchnorm/add:z:0*
T0*
_output_shapes
:G®
4batch_normalization_679/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_679_batchnorm_mul_readvariableop_resource*
_output_shapes
:G*
dtype0¼
%batch_normalization_679/batchnorm/mulMul+batch_normalization_679/batchnorm/Rsqrt:y:0<batch_normalization_679/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:G§
'batch_normalization_679/batchnorm/mul_1Muldense_753/BiasAdd:output:0)batch_normalization_679/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGª
2batch_normalization_679/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_679_batchnorm_readvariableop_1_resource*
_output_shapes
:G*
dtype0º
'batch_normalization_679/batchnorm/mul_2Mul:batch_normalization_679/batchnorm/ReadVariableOp_1:value:0)batch_normalization_679/batchnorm/mul:z:0*
T0*
_output_shapes
:Gª
2batch_normalization_679/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_679_batchnorm_readvariableop_2_resource*
_output_shapes
:G*
dtype0º
%batch_normalization_679/batchnorm/subSub:batch_normalization_679/batchnorm/ReadVariableOp_2:value:0+batch_normalization_679/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Gº
'batch_normalization_679/batchnorm/add_1AddV2+batch_normalization_679/batchnorm/mul_1:z:0)batch_normalization_679/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
leaky_re_lu_679/LeakyRelu	LeakyRelu+batch_normalization_679/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
alpha%>
dense_754/MatMul/ReadVariableOpReadVariableOp(dense_754_matmul_readvariableop_resource*
_output_shapes

:Gf*
dtype0
dense_754/MatMulMatMul'leaky_re_lu_679/LeakyRelu:activations:0'dense_754/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 dense_754/BiasAdd/ReadVariableOpReadVariableOp)dense_754_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0
dense_754/BiasAddBiasAdddense_754/MatMul:product:0(dense_754/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf¦
0batch_normalization_680/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_680_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0l
'batch_normalization_680/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_680/batchnorm/addAddV28batch_normalization_680/batchnorm/ReadVariableOp:value:00batch_normalization_680/batchnorm/add/y:output:0*
T0*
_output_shapes
:f
'batch_normalization_680/batchnorm/RsqrtRsqrt)batch_normalization_680/batchnorm/add:z:0*
T0*
_output_shapes
:f®
4batch_normalization_680/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_680_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0¼
%batch_normalization_680/batchnorm/mulMul+batch_normalization_680/batchnorm/Rsqrt:y:0<batch_normalization_680/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f§
'batch_normalization_680/batchnorm/mul_1Muldense_754/BiasAdd:output:0)batch_normalization_680/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfª
2batch_normalization_680/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_680_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0º
'batch_normalization_680/batchnorm/mul_2Mul:batch_normalization_680/batchnorm/ReadVariableOp_1:value:0)batch_normalization_680/batchnorm/mul:z:0*
T0*
_output_shapes
:fª
2batch_normalization_680/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_680_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0º
%batch_normalization_680/batchnorm/subSub:batch_normalization_680/batchnorm/ReadVariableOp_2:value:0+batch_normalization_680/batchnorm/mul_2:z:0*
T0*
_output_shapes
:fº
'batch_normalization_680/batchnorm/add_1AddV2+batch_normalization_680/batchnorm/mul_1:z:0)batch_normalization_680/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
leaky_re_lu_680/LeakyRelu	LeakyRelu+batch_normalization_680/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*
alpha%>
dense_755/MatMul/ReadVariableOpReadVariableOp(dense_755_matmul_readvariableop_resource*
_output_shapes

:f*
dtype0
dense_755/MatMulMatMul'leaky_re_lu_680/LeakyRelu:activations:0'dense_755/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_755/BiasAdd/ReadVariableOpReadVariableOp)dense_755_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_755/BiasAddBiasAdddense_755/MatMul:product:0(dense_755/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_755/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp1^batch_normalization_676/batchnorm/ReadVariableOp3^batch_normalization_676/batchnorm/ReadVariableOp_13^batch_normalization_676/batchnorm/ReadVariableOp_25^batch_normalization_676/batchnorm/mul/ReadVariableOp1^batch_normalization_677/batchnorm/ReadVariableOp3^batch_normalization_677/batchnorm/ReadVariableOp_13^batch_normalization_677/batchnorm/ReadVariableOp_25^batch_normalization_677/batchnorm/mul/ReadVariableOp1^batch_normalization_678/batchnorm/ReadVariableOp3^batch_normalization_678/batchnorm/ReadVariableOp_13^batch_normalization_678/batchnorm/ReadVariableOp_25^batch_normalization_678/batchnorm/mul/ReadVariableOp1^batch_normalization_679/batchnorm/ReadVariableOp3^batch_normalization_679/batchnorm/ReadVariableOp_13^batch_normalization_679/batchnorm/ReadVariableOp_25^batch_normalization_679/batchnorm/mul/ReadVariableOp1^batch_normalization_680/batchnorm/ReadVariableOp3^batch_normalization_680/batchnorm/ReadVariableOp_13^batch_normalization_680/batchnorm/ReadVariableOp_25^batch_normalization_680/batchnorm/mul/ReadVariableOp!^dense_750/BiasAdd/ReadVariableOp ^dense_750/MatMul/ReadVariableOp!^dense_751/BiasAdd/ReadVariableOp ^dense_751/MatMul/ReadVariableOp!^dense_752/BiasAdd/ReadVariableOp ^dense_752/MatMul/ReadVariableOp!^dense_753/BiasAdd/ReadVariableOp ^dense_753/MatMul/ReadVariableOp!^dense_754/BiasAdd/ReadVariableOp ^dense_754/MatMul/ReadVariableOp!^dense_755/BiasAdd/ReadVariableOp ^dense_755/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0batch_normalization_676/batchnorm/ReadVariableOp0batch_normalization_676/batchnorm/ReadVariableOp2h
2batch_normalization_676/batchnorm/ReadVariableOp_12batch_normalization_676/batchnorm/ReadVariableOp_12h
2batch_normalization_676/batchnorm/ReadVariableOp_22batch_normalization_676/batchnorm/ReadVariableOp_22l
4batch_normalization_676/batchnorm/mul/ReadVariableOp4batch_normalization_676/batchnorm/mul/ReadVariableOp2d
0batch_normalization_677/batchnorm/ReadVariableOp0batch_normalization_677/batchnorm/ReadVariableOp2h
2batch_normalization_677/batchnorm/ReadVariableOp_12batch_normalization_677/batchnorm/ReadVariableOp_12h
2batch_normalization_677/batchnorm/ReadVariableOp_22batch_normalization_677/batchnorm/ReadVariableOp_22l
4batch_normalization_677/batchnorm/mul/ReadVariableOp4batch_normalization_677/batchnorm/mul/ReadVariableOp2d
0batch_normalization_678/batchnorm/ReadVariableOp0batch_normalization_678/batchnorm/ReadVariableOp2h
2batch_normalization_678/batchnorm/ReadVariableOp_12batch_normalization_678/batchnorm/ReadVariableOp_12h
2batch_normalization_678/batchnorm/ReadVariableOp_22batch_normalization_678/batchnorm/ReadVariableOp_22l
4batch_normalization_678/batchnorm/mul/ReadVariableOp4batch_normalization_678/batchnorm/mul/ReadVariableOp2d
0batch_normalization_679/batchnorm/ReadVariableOp0batch_normalization_679/batchnorm/ReadVariableOp2h
2batch_normalization_679/batchnorm/ReadVariableOp_12batch_normalization_679/batchnorm/ReadVariableOp_12h
2batch_normalization_679/batchnorm/ReadVariableOp_22batch_normalization_679/batchnorm/ReadVariableOp_22l
4batch_normalization_679/batchnorm/mul/ReadVariableOp4batch_normalization_679/batchnorm/mul/ReadVariableOp2d
0batch_normalization_680/batchnorm/ReadVariableOp0batch_normalization_680/batchnorm/ReadVariableOp2h
2batch_normalization_680/batchnorm/ReadVariableOp_12batch_normalization_680/batchnorm/ReadVariableOp_12h
2batch_normalization_680/batchnorm/ReadVariableOp_22batch_normalization_680/batchnorm/ReadVariableOp_22l
4batch_normalization_680/batchnorm/mul/ReadVariableOp4batch_normalization_680/batchnorm/mul/ReadVariableOp2D
 dense_750/BiasAdd/ReadVariableOp dense_750/BiasAdd/ReadVariableOp2B
dense_750/MatMul/ReadVariableOpdense_750/MatMul/ReadVariableOp2D
 dense_751/BiasAdd/ReadVariableOp dense_751/BiasAdd/ReadVariableOp2B
dense_751/MatMul/ReadVariableOpdense_751/MatMul/ReadVariableOp2D
 dense_752/BiasAdd/ReadVariableOp dense_752/BiasAdd/ReadVariableOp2B
dense_752/MatMul/ReadVariableOpdense_752/MatMul/ReadVariableOp2D
 dense_753/BiasAdd/ReadVariableOp dense_753/BiasAdd/ReadVariableOp2B
dense_753/MatMul/ReadVariableOpdense_753/MatMul/ReadVariableOp2D
 dense_754/BiasAdd/ReadVariableOp dense_754/BiasAdd/ReadVariableOp2B
dense_754/MatMul/ReadVariableOpdense_754/MatMul/ReadVariableOp2D
 dense_755/BiasAdd/ReadVariableOp dense_755/BiasAdd/ReadVariableOp2B
dense_755/MatMul/ReadVariableOpdense_755/MatMul/ReadVariableOp:O K
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
E__inference_dense_751_layer_call_and_return_conditional_losses_750035

inputs0
matmul_readvariableop_resource:..-
biasadd_readvariableop_resource:.
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:..*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:.*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ._
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
Z
³
I__inference_sequential_74_layer_call_and_return_conditional_losses_750823
normalization_74_input
normalization_74_sub_y
normalization_74_sqrt_x"
dense_750_750742:.
dense_750_750744:.,
batch_normalization_676_750747:.,
batch_normalization_676_750749:.,
batch_normalization_676_750751:.,
batch_normalization_676_750753:."
dense_751_750757:..
dense_751_750759:.,
batch_normalization_677_750762:.,
batch_normalization_677_750764:.,
batch_normalization_677_750766:.,
batch_normalization_677_750768:."
dense_752_750772:.G
dense_752_750774:G,
batch_normalization_678_750777:G,
batch_normalization_678_750779:G,
batch_normalization_678_750781:G,
batch_normalization_678_750783:G"
dense_753_750787:GG
dense_753_750789:G,
batch_normalization_679_750792:G,
batch_normalization_679_750794:G,
batch_normalization_679_750796:G,
batch_normalization_679_750798:G"
dense_754_750802:Gf
dense_754_750804:f,
batch_normalization_680_750807:f,
batch_normalization_680_750809:f,
batch_normalization_680_750811:f,
batch_normalization_680_750813:f"
dense_755_750817:f
dense_755_750819:
identity¢/batch_normalization_676/StatefulPartitionedCall¢/batch_normalization_677/StatefulPartitionedCall¢/batch_normalization_678/StatefulPartitionedCall¢/batch_normalization_679/StatefulPartitionedCall¢/batch_normalization_680/StatefulPartitionedCall¢!dense_750/StatefulPartitionedCall¢!dense_751/StatefulPartitionedCall¢!dense_752/StatefulPartitionedCall¢!dense_753/StatefulPartitionedCall¢!dense_754/StatefulPartitionedCall¢!dense_755/StatefulPartitionedCall}
normalization_74/subSubnormalization_74_inputnormalization_74_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_74/SqrtSqrtnormalization_74_sqrt_x*
T0*
_output_shapes

:_
normalization_74/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_74/MaximumMaximumnormalization_74/Sqrt:y:0#normalization_74/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_74/truedivRealDivnormalization_74/sub:z:0normalization_74/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_750/StatefulPartitionedCallStatefulPartitionedCallnormalization_74/truediv:z:0dense_750_750742dense_750_750744*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_750_layer_call_and_return_conditional_losses_750003
/batch_normalization_676/StatefulPartitionedCallStatefulPartitionedCall*dense_750/StatefulPartitionedCall:output:0batch_normalization_676_750747batch_normalization_676_750749batch_normalization_676_750751batch_normalization_676_750753*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_676_layer_call_and_return_conditional_losses_749640ø
leaky_re_lu_676/PartitionedCallPartitionedCall8batch_normalization_676/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_676_layer_call_and_return_conditional_losses_750023
!dense_751/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_676/PartitionedCall:output:0dense_751_750757dense_751_750759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_751_layer_call_and_return_conditional_losses_750035
/batch_normalization_677/StatefulPartitionedCallStatefulPartitionedCall*dense_751/StatefulPartitionedCall:output:0batch_normalization_677_750762batch_normalization_677_750764batch_normalization_677_750766batch_normalization_677_750768*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_677_layer_call_and_return_conditional_losses_749722ø
leaky_re_lu_677/PartitionedCallPartitionedCall8batch_normalization_677/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_677_layer_call_and_return_conditional_losses_750055
!dense_752/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_677/PartitionedCall:output:0dense_752_750772dense_752_750774*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_752_layer_call_and_return_conditional_losses_750067
/batch_normalization_678/StatefulPartitionedCallStatefulPartitionedCall*dense_752/StatefulPartitionedCall:output:0batch_normalization_678_750777batch_normalization_678_750779batch_normalization_678_750781batch_normalization_678_750783*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_678_layer_call_and_return_conditional_losses_749804ø
leaky_re_lu_678/PartitionedCallPartitionedCall8batch_normalization_678/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_678_layer_call_and_return_conditional_losses_750087
!dense_753/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_678/PartitionedCall:output:0dense_753_750787dense_753_750789*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_753_layer_call_and_return_conditional_losses_750099
/batch_normalization_679/StatefulPartitionedCallStatefulPartitionedCall*dense_753/StatefulPartitionedCall:output:0batch_normalization_679_750792batch_normalization_679_750794batch_normalization_679_750796batch_normalization_679_750798*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_679_layer_call_and_return_conditional_losses_749886ø
leaky_re_lu_679/PartitionedCallPartitionedCall8batch_normalization_679/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_679_layer_call_and_return_conditional_losses_750119
!dense_754/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_679/PartitionedCall:output:0dense_754_750802dense_754_750804*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_754_layer_call_and_return_conditional_losses_750131
/batch_normalization_680/StatefulPartitionedCallStatefulPartitionedCall*dense_754/StatefulPartitionedCall:output:0batch_normalization_680_750807batch_normalization_680_750809batch_normalization_680_750811batch_normalization_680_750813*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_680_layer_call_and_return_conditional_losses_749968ø
leaky_re_lu_680/PartitionedCallPartitionedCall8batch_normalization_680/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_680_layer_call_and_return_conditional_losses_750151
!dense_755/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_680/PartitionedCall:output:0dense_755_750817dense_755_750819*
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
E__inference_dense_755_layer_call_and_return_conditional_losses_750163y
IdentityIdentity*dense_755/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_676/StatefulPartitionedCall0^batch_normalization_677/StatefulPartitionedCall0^batch_normalization_678/StatefulPartitionedCall0^batch_normalization_679/StatefulPartitionedCall0^batch_normalization_680/StatefulPartitionedCall"^dense_750/StatefulPartitionedCall"^dense_751/StatefulPartitionedCall"^dense_752/StatefulPartitionedCall"^dense_753/StatefulPartitionedCall"^dense_754/StatefulPartitionedCall"^dense_755/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_676/StatefulPartitionedCall/batch_normalization_676/StatefulPartitionedCall2b
/batch_normalization_677/StatefulPartitionedCall/batch_normalization_677/StatefulPartitionedCall2b
/batch_normalization_678/StatefulPartitionedCall/batch_normalization_678/StatefulPartitionedCall2b
/batch_normalization_679/StatefulPartitionedCall/batch_normalization_679/StatefulPartitionedCall2b
/batch_normalization_680/StatefulPartitionedCall/batch_normalization_680/StatefulPartitionedCall2F
!dense_750/StatefulPartitionedCall!dense_750/StatefulPartitionedCall2F
!dense_751/StatefulPartitionedCall!dense_751/StatefulPartitionedCall2F
!dense_752/StatefulPartitionedCall!dense_752/StatefulPartitionedCall2F
!dense_753/StatefulPartitionedCall!dense_753/StatefulPartitionedCall2F
!dense_754/StatefulPartitionedCall!dense_754/StatefulPartitionedCall2F
!dense_755/StatefulPartitionedCall!dense_755/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_74_input:$ 

_output_shapes

::$ 

_output_shapes

:
%
ì
S__inference_batch_normalization_678_layer_call_and_return_conditional_losses_751746

inputs5
'assignmovingavg_readvariableop_resource:G7
)assignmovingavg_1_readvariableop_resource:G3
%batchnorm_mul_readvariableop_resource:G/
!batchnorm_readvariableop_resource:G
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:G*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:G
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:G*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:G*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:G*
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
:G*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Gx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:G¬
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
:G*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:G~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:G´
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
:GP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:G~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:G*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Gc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Gv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:G*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Gr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_678_layer_call_fn_751679

inputs
unknown:G
	unknown_0:G
	unknown_1:G
	unknown_2:G
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_678_layer_call_and_return_conditional_losses_749757o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
©Ø
þ7
"__inference__traced_restore_752538
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 5
#assignvariableop_3_dense_750_kernel:./
!assignvariableop_4_dense_750_bias:.>
0assignvariableop_5_batch_normalization_676_gamma:.=
/assignvariableop_6_batch_normalization_676_beta:.D
6assignvariableop_7_batch_normalization_676_moving_mean:.H
:assignvariableop_8_batch_normalization_676_moving_variance:.5
#assignvariableop_9_dense_751_kernel:..0
"assignvariableop_10_dense_751_bias:.?
1assignvariableop_11_batch_normalization_677_gamma:.>
0assignvariableop_12_batch_normalization_677_beta:.E
7assignvariableop_13_batch_normalization_677_moving_mean:.I
;assignvariableop_14_batch_normalization_677_moving_variance:.6
$assignvariableop_15_dense_752_kernel:.G0
"assignvariableop_16_dense_752_bias:G?
1assignvariableop_17_batch_normalization_678_gamma:G>
0assignvariableop_18_batch_normalization_678_beta:GE
7assignvariableop_19_batch_normalization_678_moving_mean:GI
;assignvariableop_20_batch_normalization_678_moving_variance:G6
$assignvariableop_21_dense_753_kernel:GG0
"assignvariableop_22_dense_753_bias:G?
1assignvariableop_23_batch_normalization_679_gamma:G>
0assignvariableop_24_batch_normalization_679_beta:GE
7assignvariableop_25_batch_normalization_679_moving_mean:GI
;assignvariableop_26_batch_normalization_679_moving_variance:G6
$assignvariableop_27_dense_754_kernel:Gf0
"assignvariableop_28_dense_754_bias:f?
1assignvariableop_29_batch_normalization_680_gamma:f>
0assignvariableop_30_batch_normalization_680_beta:fE
7assignvariableop_31_batch_normalization_680_moving_mean:fI
;assignvariableop_32_batch_normalization_680_moving_variance:f6
$assignvariableop_33_dense_755_kernel:f0
"assignvariableop_34_dense_755_bias:'
assignvariableop_35_adam_iter:	 )
assignvariableop_36_adam_beta_1: )
assignvariableop_37_adam_beta_2: (
assignvariableop_38_adam_decay: #
assignvariableop_39_total: %
assignvariableop_40_count_1: =
+assignvariableop_41_adam_dense_750_kernel_m:.7
)assignvariableop_42_adam_dense_750_bias_m:.F
8assignvariableop_43_adam_batch_normalization_676_gamma_m:.E
7assignvariableop_44_adam_batch_normalization_676_beta_m:.=
+assignvariableop_45_adam_dense_751_kernel_m:..7
)assignvariableop_46_adam_dense_751_bias_m:.F
8assignvariableop_47_adam_batch_normalization_677_gamma_m:.E
7assignvariableop_48_adam_batch_normalization_677_beta_m:.=
+assignvariableop_49_adam_dense_752_kernel_m:.G7
)assignvariableop_50_adam_dense_752_bias_m:GF
8assignvariableop_51_adam_batch_normalization_678_gamma_m:GE
7assignvariableop_52_adam_batch_normalization_678_beta_m:G=
+assignvariableop_53_adam_dense_753_kernel_m:GG7
)assignvariableop_54_adam_dense_753_bias_m:GF
8assignvariableop_55_adam_batch_normalization_679_gamma_m:GE
7assignvariableop_56_adam_batch_normalization_679_beta_m:G=
+assignvariableop_57_adam_dense_754_kernel_m:Gf7
)assignvariableop_58_adam_dense_754_bias_m:fF
8assignvariableop_59_adam_batch_normalization_680_gamma_m:fE
7assignvariableop_60_adam_batch_normalization_680_beta_m:f=
+assignvariableop_61_adam_dense_755_kernel_m:f7
)assignvariableop_62_adam_dense_755_bias_m:=
+assignvariableop_63_adam_dense_750_kernel_v:.7
)assignvariableop_64_adam_dense_750_bias_v:.F
8assignvariableop_65_adam_batch_normalization_676_gamma_v:.E
7assignvariableop_66_adam_batch_normalization_676_beta_v:.=
+assignvariableop_67_adam_dense_751_kernel_v:..7
)assignvariableop_68_adam_dense_751_bias_v:.F
8assignvariableop_69_adam_batch_normalization_677_gamma_v:.E
7assignvariableop_70_adam_batch_normalization_677_beta_v:.=
+assignvariableop_71_adam_dense_752_kernel_v:.G7
)assignvariableop_72_adam_dense_752_bias_v:GF
8assignvariableop_73_adam_batch_normalization_678_gamma_v:GE
7assignvariableop_74_adam_batch_normalization_678_beta_v:G=
+assignvariableop_75_adam_dense_753_kernel_v:GG7
)assignvariableop_76_adam_dense_753_bias_v:GF
8assignvariableop_77_adam_batch_normalization_679_gamma_v:GE
7assignvariableop_78_adam_batch_normalization_679_beta_v:G=
+assignvariableop_79_adam_dense_754_kernel_v:Gf7
)assignvariableop_80_adam_dense_754_bias_v:fF
8assignvariableop_81_adam_batch_normalization_680_gamma_v:fE
7assignvariableop_82_adam_batch_normalization_680_beta_v:f=
+assignvariableop_83_adam_dense_755_kernel_v:f7
)assignvariableop_84_adam_dense_755_bias_v:
identity_86¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_9Ä/
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*ê.
valueà.BÝ.VB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*Á
value·B´VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ï
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*î
_output_shapesÛ
Ø::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*d
dtypesZ
X2V		[
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
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_750_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_750_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp0assignvariableop_5_batch_normalization_676_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_676_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_676_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_8AssignVariableOp:assignvariableop_8_batch_normalization_676_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_751_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_751_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_11AssignVariableOp1assignvariableop_11_batch_normalization_677_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_677_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_677_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_14AssignVariableOp;assignvariableop_14_batch_normalization_677_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_752_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_752_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_17AssignVariableOp1assignvariableop_17_batch_normalization_678_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_678_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_19AssignVariableOp7assignvariableop_19_batch_normalization_678_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_20AssignVariableOp;assignvariableop_20_batch_normalization_678_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp$assignvariableop_21_dense_753_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_753_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_23AssignVariableOp1assignvariableop_23_batch_normalization_679_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_679_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_25AssignVariableOp7assignvariableop_25_batch_normalization_679_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_26AssignVariableOp;assignvariableop_26_batch_normalization_679_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp$assignvariableop_27_dense_754_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_754_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_29AssignVariableOp1assignvariableop_29_batch_normalization_680_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_normalization_680_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_31AssignVariableOp7assignvariableop_31_batch_normalization_680_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_32AssignVariableOp;assignvariableop_32_batch_normalization_680_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp$assignvariableop_33_dense_755_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_755_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_iterIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOpassignvariableop_36_adam_beta_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOpassignvariableop_37_adam_beta_2Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOpassignvariableop_38_adam_decayIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOpassignvariableop_39_totalIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_dense_750_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_dense_750_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_43AssignVariableOp8assignvariableop_43_adam_batch_normalization_676_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_44AssignVariableOp7assignvariableop_44_adam_batch_normalization_676_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_dense_751_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_dense_751_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_47AssignVariableOp8assignvariableop_47_adam_batch_normalization_677_gamma_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_48AssignVariableOp7assignvariableop_48_adam_batch_normalization_677_beta_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_dense_752_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_dense_752_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_51AssignVariableOp8assignvariableop_51_adam_batch_normalization_678_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_52AssignVariableOp7assignvariableop_52_adam_batch_normalization_678_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_dense_753_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_dense_753_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_55AssignVariableOp8assignvariableop_55_adam_batch_normalization_679_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_56AssignVariableOp7assignvariableop_56_adam_batch_normalization_679_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_dense_754_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_dense_754_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_59AssignVariableOp8assignvariableop_59_adam_batch_normalization_680_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_60AssignVariableOp7assignvariableop_60_adam_batch_normalization_680_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_dense_755_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_dense_755_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_dense_750_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_dense_750_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_65AssignVariableOp8assignvariableop_65_adam_batch_normalization_676_gamma_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_66AssignVariableOp7assignvariableop_66_adam_batch_normalization_676_beta_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_dense_751_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_dense_751_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_69AssignVariableOp8assignvariableop_69_adam_batch_normalization_677_gamma_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_70AssignVariableOp7assignvariableop_70_adam_batch_normalization_677_beta_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_dense_752_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_dense_752_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_73AssignVariableOp8assignvariableop_73_adam_batch_normalization_678_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_74AssignVariableOp7assignvariableop_74_adam_batch_normalization_678_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_dense_753_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_dense_753_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_77AssignVariableOp8assignvariableop_77_adam_batch_normalization_679_gamma_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_78AssignVariableOp7assignvariableop_78_adam_batch_normalization_679_beta_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_79AssignVariableOp+assignvariableop_79_adam_dense_754_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_80AssignVariableOp)assignvariableop_80_adam_dense_754_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_81AssignVariableOp8assignvariableop_81_adam_batch_normalization_680_gamma_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_82AssignVariableOp7assignvariableop_82_adam_batch_normalization_680_beta_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_83AssignVariableOp+assignvariableop_83_adam_dense_755_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_84AssignVariableOp)assignvariableop_84_adam_dense_755_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_85Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_86IdentityIdentity_85:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_86Identity_86:output:0*Á
_input_shapes¯
¬: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_84AssignVariableOp_842(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
È	
ö
E__inference_dense_754_layer_call_and_return_conditional_losses_750131

inputs0
matmul_readvariableop_resource:Gf-
biasadd_readvariableop_resource:f
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Gf*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:f*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
È	
ö
E__inference_dense_750_layer_call_and_return_conditional_losses_750003

inputs0
matmul_readvariableop_resource:.-
biasadd_readvariableop_resource:.
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:.*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:.*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ._
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.w
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
©¾
"
I__inference_sequential_74_layer_call_and_return_conditional_losses_751307

inputs
normalization_74_sub_y
normalization_74_sqrt_x:
(dense_750_matmul_readvariableop_resource:.7
)dense_750_biasadd_readvariableop_resource:.M
?batch_normalization_676_assignmovingavg_readvariableop_resource:.O
Abatch_normalization_676_assignmovingavg_1_readvariableop_resource:.K
=batch_normalization_676_batchnorm_mul_readvariableop_resource:.G
9batch_normalization_676_batchnorm_readvariableop_resource:.:
(dense_751_matmul_readvariableop_resource:..7
)dense_751_biasadd_readvariableop_resource:.M
?batch_normalization_677_assignmovingavg_readvariableop_resource:.O
Abatch_normalization_677_assignmovingavg_1_readvariableop_resource:.K
=batch_normalization_677_batchnorm_mul_readvariableop_resource:.G
9batch_normalization_677_batchnorm_readvariableop_resource:.:
(dense_752_matmul_readvariableop_resource:.G7
)dense_752_biasadd_readvariableop_resource:GM
?batch_normalization_678_assignmovingavg_readvariableop_resource:GO
Abatch_normalization_678_assignmovingavg_1_readvariableop_resource:GK
=batch_normalization_678_batchnorm_mul_readvariableop_resource:GG
9batch_normalization_678_batchnorm_readvariableop_resource:G:
(dense_753_matmul_readvariableop_resource:GG7
)dense_753_biasadd_readvariableop_resource:GM
?batch_normalization_679_assignmovingavg_readvariableop_resource:GO
Abatch_normalization_679_assignmovingavg_1_readvariableop_resource:GK
=batch_normalization_679_batchnorm_mul_readvariableop_resource:GG
9batch_normalization_679_batchnorm_readvariableop_resource:G:
(dense_754_matmul_readvariableop_resource:Gf7
)dense_754_biasadd_readvariableop_resource:fM
?batch_normalization_680_assignmovingavg_readvariableop_resource:fO
Abatch_normalization_680_assignmovingavg_1_readvariableop_resource:fK
=batch_normalization_680_batchnorm_mul_readvariableop_resource:fG
9batch_normalization_680_batchnorm_readvariableop_resource:f:
(dense_755_matmul_readvariableop_resource:f7
)dense_755_biasadd_readvariableop_resource:
identity¢'batch_normalization_676/AssignMovingAvg¢6batch_normalization_676/AssignMovingAvg/ReadVariableOp¢)batch_normalization_676/AssignMovingAvg_1¢8batch_normalization_676/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_676/batchnorm/ReadVariableOp¢4batch_normalization_676/batchnorm/mul/ReadVariableOp¢'batch_normalization_677/AssignMovingAvg¢6batch_normalization_677/AssignMovingAvg/ReadVariableOp¢)batch_normalization_677/AssignMovingAvg_1¢8batch_normalization_677/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_677/batchnorm/ReadVariableOp¢4batch_normalization_677/batchnorm/mul/ReadVariableOp¢'batch_normalization_678/AssignMovingAvg¢6batch_normalization_678/AssignMovingAvg/ReadVariableOp¢)batch_normalization_678/AssignMovingAvg_1¢8batch_normalization_678/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_678/batchnorm/ReadVariableOp¢4batch_normalization_678/batchnorm/mul/ReadVariableOp¢'batch_normalization_679/AssignMovingAvg¢6batch_normalization_679/AssignMovingAvg/ReadVariableOp¢)batch_normalization_679/AssignMovingAvg_1¢8batch_normalization_679/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_679/batchnorm/ReadVariableOp¢4batch_normalization_679/batchnorm/mul/ReadVariableOp¢'batch_normalization_680/AssignMovingAvg¢6batch_normalization_680/AssignMovingAvg/ReadVariableOp¢)batch_normalization_680/AssignMovingAvg_1¢8batch_normalization_680/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_680/batchnorm/ReadVariableOp¢4batch_normalization_680/batchnorm/mul/ReadVariableOp¢ dense_750/BiasAdd/ReadVariableOp¢dense_750/MatMul/ReadVariableOp¢ dense_751/BiasAdd/ReadVariableOp¢dense_751/MatMul/ReadVariableOp¢ dense_752/BiasAdd/ReadVariableOp¢dense_752/MatMul/ReadVariableOp¢ dense_753/BiasAdd/ReadVariableOp¢dense_753/MatMul/ReadVariableOp¢ dense_754/BiasAdd/ReadVariableOp¢dense_754/MatMul/ReadVariableOp¢ dense_755/BiasAdd/ReadVariableOp¢dense_755/MatMul/ReadVariableOpm
normalization_74/subSubinputsnormalization_74_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_74/SqrtSqrtnormalization_74_sqrt_x*
T0*
_output_shapes

:_
normalization_74/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_74/MaximumMaximumnormalization_74/Sqrt:y:0#normalization_74/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_74/truedivRealDivnormalization_74/sub:z:0normalization_74/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_750/MatMul/ReadVariableOpReadVariableOp(dense_750_matmul_readvariableop_resource*
_output_shapes

:.*
dtype0
dense_750/MatMulMatMulnormalization_74/truediv:z:0'dense_750/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 dense_750/BiasAdd/ReadVariableOpReadVariableOp)dense_750_biasadd_readvariableop_resource*
_output_shapes
:.*
dtype0
dense_750/BiasAddBiasAdddense_750/MatMul:product:0(dense_750/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
6batch_normalization_676/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_676/moments/meanMeandense_750/BiasAdd:output:0?batch_normalization_676/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(
,batch_normalization_676/moments/StopGradientStopGradient-batch_normalization_676/moments/mean:output:0*
T0*
_output_shapes

:.Ë
1batch_normalization_676/moments/SquaredDifferenceSquaredDifferencedense_750/BiasAdd:output:05batch_normalization_676/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
:batch_normalization_676/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_676/moments/varianceMean5batch_normalization_676/moments/SquaredDifference:z:0Cbatch_normalization_676/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(
'batch_normalization_676/moments/SqueezeSqueeze-batch_normalization_676/moments/mean:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 £
)batch_normalization_676/moments/Squeeze_1Squeeze1batch_normalization_676/moments/variance:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 r
-batch_normalization_676/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_676/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_676_assignmovingavg_readvariableop_resource*
_output_shapes
:.*
dtype0É
+batch_normalization_676/AssignMovingAvg/subSub>batch_normalization_676/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_676/moments/Squeeze:output:0*
T0*
_output_shapes
:.À
+batch_normalization_676/AssignMovingAvg/mulMul/batch_normalization_676/AssignMovingAvg/sub:z:06batch_normalization_676/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:.
'batch_normalization_676/AssignMovingAvgAssignSubVariableOp?batch_normalization_676_assignmovingavg_readvariableop_resource/batch_normalization_676/AssignMovingAvg/mul:z:07^batch_normalization_676/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_676/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_676/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_676_assignmovingavg_1_readvariableop_resource*
_output_shapes
:.*
dtype0Ï
-batch_normalization_676/AssignMovingAvg_1/subSub@batch_normalization_676/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_676/moments/Squeeze_1:output:0*
T0*
_output_shapes
:.Æ
-batch_normalization_676/AssignMovingAvg_1/mulMul1batch_normalization_676/AssignMovingAvg_1/sub:z:08batch_normalization_676/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:.
)batch_normalization_676/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_676_assignmovingavg_1_readvariableop_resource1batch_normalization_676/AssignMovingAvg_1/mul:z:09^batch_normalization_676/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_676/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_676/batchnorm/addAddV22batch_normalization_676/moments/Squeeze_1:output:00batch_normalization_676/batchnorm/add/y:output:0*
T0*
_output_shapes
:.
'batch_normalization_676/batchnorm/RsqrtRsqrt)batch_normalization_676/batchnorm/add:z:0*
T0*
_output_shapes
:.®
4batch_normalization_676/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_676_batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0¼
%batch_normalization_676/batchnorm/mulMul+batch_normalization_676/batchnorm/Rsqrt:y:0<batch_normalization_676/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.§
'batch_normalization_676/batchnorm/mul_1Muldense_750/BiasAdd:output:0)batch_normalization_676/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.°
'batch_normalization_676/batchnorm/mul_2Mul0batch_normalization_676/moments/Squeeze:output:0)batch_normalization_676/batchnorm/mul:z:0*
T0*
_output_shapes
:.¦
0batch_normalization_676/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_676_batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0¸
%batch_normalization_676/batchnorm/subSub8batch_normalization_676/batchnorm/ReadVariableOp:value:0+batch_normalization_676/batchnorm/mul_2:z:0*
T0*
_output_shapes
:.º
'batch_normalization_676/batchnorm/add_1AddV2+batch_normalization_676/batchnorm/mul_1:z:0)batch_normalization_676/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
leaky_re_lu_676/LeakyRelu	LeakyRelu+batch_normalization_676/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
alpha%>
dense_751/MatMul/ReadVariableOpReadVariableOp(dense_751_matmul_readvariableop_resource*
_output_shapes

:..*
dtype0
dense_751/MatMulMatMul'leaky_re_lu_676/LeakyRelu:activations:0'dense_751/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 dense_751/BiasAdd/ReadVariableOpReadVariableOp)dense_751_biasadd_readvariableop_resource*
_output_shapes
:.*
dtype0
dense_751/BiasAddBiasAdddense_751/MatMul:product:0(dense_751/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
6batch_normalization_677/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_677/moments/meanMeandense_751/BiasAdd:output:0?batch_normalization_677/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(
,batch_normalization_677/moments/StopGradientStopGradient-batch_normalization_677/moments/mean:output:0*
T0*
_output_shapes

:.Ë
1batch_normalization_677/moments/SquaredDifferenceSquaredDifferencedense_751/BiasAdd:output:05batch_normalization_677/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
:batch_normalization_677/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_677/moments/varianceMean5batch_normalization_677/moments/SquaredDifference:z:0Cbatch_normalization_677/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:.*
	keep_dims(
'batch_normalization_677/moments/SqueezeSqueeze-batch_normalization_677/moments/mean:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 £
)batch_normalization_677/moments/Squeeze_1Squeeze1batch_normalization_677/moments/variance:output:0*
T0*
_output_shapes
:.*
squeeze_dims
 r
-batch_normalization_677/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_677/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_677_assignmovingavg_readvariableop_resource*
_output_shapes
:.*
dtype0É
+batch_normalization_677/AssignMovingAvg/subSub>batch_normalization_677/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_677/moments/Squeeze:output:0*
T0*
_output_shapes
:.À
+batch_normalization_677/AssignMovingAvg/mulMul/batch_normalization_677/AssignMovingAvg/sub:z:06batch_normalization_677/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:.
'batch_normalization_677/AssignMovingAvgAssignSubVariableOp?batch_normalization_677_assignmovingavg_readvariableop_resource/batch_normalization_677/AssignMovingAvg/mul:z:07^batch_normalization_677/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_677/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_677/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_677_assignmovingavg_1_readvariableop_resource*
_output_shapes
:.*
dtype0Ï
-batch_normalization_677/AssignMovingAvg_1/subSub@batch_normalization_677/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_677/moments/Squeeze_1:output:0*
T0*
_output_shapes
:.Æ
-batch_normalization_677/AssignMovingAvg_1/mulMul1batch_normalization_677/AssignMovingAvg_1/sub:z:08batch_normalization_677/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:.
)batch_normalization_677/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_677_assignmovingavg_1_readvariableop_resource1batch_normalization_677/AssignMovingAvg_1/mul:z:09^batch_normalization_677/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_677/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_677/batchnorm/addAddV22batch_normalization_677/moments/Squeeze_1:output:00batch_normalization_677/batchnorm/add/y:output:0*
T0*
_output_shapes
:.
'batch_normalization_677/batchnorm/RsqrtRsqrt)batch_normalization_677/batchnorm/add:z:0*
T0*
_output_shapes
:.®
4batch_normalization_677/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_677_batchnorm_mul_readvariableop_resource*
_output_shapes
:.*
dtype0¼
%batch_normalization_677/batchnorm/mulMul+batch_normalization_677/batchnorm/Rsqrt:y:0<batch_normalization_677/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:.§
'batch_normalization_677/batchnorm/mul_1Muldense_751/BiasAdd:output:0)batch_normalization_677/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.°
'batch_normalization_677/batchnorm/mul_2Mul0batch_normalization_677/moments/Squeeze:output:0)batch_normalization_677/batchnorm/mul:z:0*
T0*
_output_shapes
:.¦
0batch_normalization_677/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_677_batchnorm_readvariableop_resource*
_output_shapes
:.*
dtype0¸
%batch_normalization_677/batchnorm/subSub8batch_normalization_677/batchnorm/ReadVariableOp:value:0+batch_normalization_677/batchnorm/mul_2:z:0*
T0*
_output_shapes
:.º
'batch_normalization_677/batchnorm/add_1AddV2+batch_normalization_677/batchnorm/mul_1:z:0)batch_normalization_677/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
leaky_re_lu_677/LeakyRelu	LeakyRelu+batch_normalization_677/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
alpha%>
dense_752/MatMul/ReadVariableOpReadVariableOp(dense_752_matmul_readvariableop_resource*
_output_shapes

:.G*
dtype0
dense_752/MatMulMatMul'leaky_re_lu_677/LeakyRelu:activations:0'dense_752/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 dense_752/BiasAdd/ReadVariableOpReadVariableOp)dense_752_biasadd_readvariableop_resource*
_output_shapes
:G*
dtype0
dense_752/BiasAddBiasAdddense_752/MatMul:product:0(dense_752/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
6batch_normalization_678/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_678/moments/meanMeandense_752/BiasAdd:output:0?batch_normalization_678/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:G*
	keep_dims(
,batch_normalization_678/moments/StopGradientStopGradient-batch_normalization_678/moments/mean:output:0*
T0*
_output_shapes

:GË
1batch_normalization_678/moments/SquaredDifferenceSquaredDifferencedense_752/BiasAdd:output:05batch_normalization_678/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
:batch_normalization_678/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_678/moments/varianceMean5batch_normalization_678/moments/SquaredDifference:z:0Cbatch_normalization_678/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:G*
	keep_dims(
'batch_normalization_678/moments/SqueezeSqueeze-batch_normalization_678/moments/mean:output:0*
T0*
_output_shapes
:G*
squeeze_dims
 £
)batch_normalization_678/moments/Squeeze_1Squeeze1batch_normalization_678/moments/variance:output:0*
T0*
_output_shapes
:G*
squeeze_dims
 r
-batch_normalization_678/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_678/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_678_assignmovingavg_readvariableop_resource*
_output_shapes
:G*
dtype0É
+batch_normalization_678/AssignMovingAvg/subSub>batch_normalization_678/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_678/moments/Squeeze:output:0*
T0*
_output_shapes
:GÀ
+batch_normalization_678/AssignMovingAvg/mulMul/batch_normalization_678/AssignMovingAvg/sub:z:06batch_normalization_678/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:G
'batch_normalization_678/AssignMovingAvgAssignSubVariableOp?batch_normalization_678_assignmovingavg_readvariableop_resource/batch_normalization_678/AssignMovingAvg/mul:z:07^batch_normalization_678/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_678/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_678/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_678_assignmovingavg_1_readvariableop_resource*
_output_shapes
:G*
dtype0Ï
-batch_normalization_678/AssignMovingAvg_1/subSub@batch_normalization_678/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_678/moments/Squeeze_1:output:0*
T0*
_output_shapes
:GÆ
-batch_normalization_678/AssignMovingAvg_1/mulMul1batch_normalization_678/AssignMovingAvg_1/sub:z:08batch_normalization_678/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:G
)batch_normalization_678/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_678_assignmovingavg_1_readvariableop_resource1batch_normalization_678/AssignMovingAvg_1/mul:z:09^batch_normalization_678/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_678/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_678/batchnorm/addAddV22batch_normalization_678/moments/Squeeze_1:output:00batch_normalization_678/batchnorm/add/y:output:0*
T0*
_output_shapes
:G
'batch_normalization_678/batchnorm/RsqrtRsqrt)batch_normalization_678/batchnorm/add:z:0*
T0*
_output_shapes
:G®
4batch_normalization_678/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_678_batchnorm_mul_readvariableop_resource*
_output_shapes
:G*
dtype0¼
%batch_normalization_678/batchnorm/mulMul+batch_normalization_678/batchnorm/Rsqrt:y:0<batch_normalization_678/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:G§
'batch_normalization_678/batchnorm/mul_1Muldense_752/BiasAdd:output:0)batch_normalization_678/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG°
'batch_normalization_678/batchnorm/mul_2Mul0batch_normalization_678/moments/Squeeze:output:0)batch_normalization_678/batchnorm/mul:z:0*
T0*
_output_shapes
:G¦
0batch_normalization_678/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_678_batchnorm_readvariableop_resource*
_output_shapes
:G*
dtype0¸
%batch_normalization_678/batchnorm/subSub8batch_normalization_678/batchnorm/ReadVariableOp:value:0+batch_normalization_678/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Gº
'batch_normalization_678/batchnorm/add_1AddV2+batch_normalization_678/batchnorm/mul_1:z:0)batch_normalization_678/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
leaky_re_lu_678/LeakyRelu	LeakyRelu+batch_normalization_678/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
alpha%>
dense_753/MatMul/ReadVariableOpReadVariableOp(dense_753_matmul_readvariableop_resource*
_output_shapes

:GG*
dtype0
dense_753/MatMulMatMul'leaky_re_lu_678/LeakyRelu:activations:0'dense_753/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 dense_753/BiasAdd/ReadVariableOpReadVariableOp)dense_753_biasadd_readvariableop_resource*
_output_shapes
:G*
dtype0
dense_753/BiasAddBiasAdddense_753/MatMul:product:0(dense_753/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
6batch_normalization_679/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_679/moments/meanMeandense_753/BiasAdd:output:0?batch_normalization_679/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:G*
	keep_dims(
,batch_normalization_679/moments/StopGradientStopGradient-batch_normalization_679/moments/mean:output:0*
T0*
_output_shapes

:GË
1batch_normalization_679/moments/SquaredDifferenceSquaredDifferencedense_753/BiasAdd:output:05batch_normalization_679/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
:batch_normalization_679/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_679/moments/varianceMean5batch_normalization_679/moments/SquaredDifference:z:0Cbatch_normalization_679/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:G*
	keep_dims(
'batch_normalization_679/moments/SqueezeSqueeze-batch_normalization_679/moments/mean:output:0*
T0*
_output_shapes
:G*
squeeze_dims
 £
)batch_normalization_679/moments/Squeeze_1Squeeze1batch_normalization_679/moments/variance:output:0*
T0*
_output_shapes
:G*
squeeze_dims
 r
-batch_normalization_679/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_679/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_679_assignmovingavg_readvariableop_resource*
_output_shapes
:G*
dtype0É
+batch_normalization_679/AssignMovingAvg/subSub>batch_normalization_679/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_679/moments/Squeeze:output:0*
T0*
_output_shapes
:GÀ
+batch_normalization_679/AssignMovingAvg/mulMul/batch_normalization_679/AssignMovingAvg/sub:z:06batch_normalization_679/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:G
'batch_normalization_679/AssignMovingAvgAssignSubVariableOp?batch_normalization_679_assignmovingavg_readvariableop_resource/batch_normalization_679/AssignMovingAvg/mul:z:07^batch_normalization_679/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_679/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_679/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_679_assignmovingavg_1_readvariableop_resource*
_output_shapes
:G*
dtype0Ï
-batch_normalization_679/AssignMovingAvg_1/subSub@batch_normalization_679/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_679/moments/Squeeze_1:output:0*
T0*
_output_shapes
:GÆ
-batch_normalization_679/AssignMovingAvg_1/mulMul1batch_normalization_679/AssignMovingAvg_1/sub:z:08batch_normalization_679/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:G
)batch_normalization_679/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_679_assignmovingavg_1_readvariableop_resource1batch_normalization_679/AssignMovingAvg_1/mul:z:09^batch_normalization_679/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_679/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_679/batchnorm/addAddV22batch_normalization_679/moments/Squeeze_1:output:00batch_normalization_679/batchnorm/add/y:output:0*
T0*
_output_shapes
:G
'batch_normalization_679/batchnorm/RsqrtRsqrt)batch_normalization_679/batchnorm/add:z:0*
T0*
_output_shapes
:G®
4batch_normalization_679/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_679_batchnorm_mul_readvariableop_resource*
_output_shapes
:G*
dtype0¼
%batch_normalization_679/batchnorm/mulMul+batch_normalization_679/batchnorm/Rsqrt:y:0<batch_normalization_679/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:G§
'batch_normalization_679/batchnorm/mul_1Muldense_753/BiasAdd:output:0)batch_normalization_679/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG°
'batch_normalization_679/batchnorm/mul_2Mul0batch_normalization_679/moments/Squeeze:output:0)batch_normalization_679/batchnorm/mul:z:0*
T0*
_output_shapes
:G¦
0batch_normalization_679/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_679_batchnorm_readvariableop_resource*
_output_shapes
:G*
dtype0¸
%batch_normalization_679/batchnorm/subSub8batch_normalization_679/batchnorm/ReadVariableOp:value:0+batch_normalization_679/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Gº
'batch_normalization_679/batchnorm/add_1AddV2+batch_normalization_679/batchnorm/mul_1:z:0)batch_normalization_679/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
leaky_re_lu_679/LeakyRelu	LeakyRelu+batch_normalization_679/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
alpha%>
dense_754/MatMul/ReadVariableOpReadVariableOp(dense_754_matmul_readvariableop_resource*
_output_shapes

:Gf*
dtype0
dense_754/MatMulMatMul'leaky_re_lu_679/LeakyRelu:activations:0'dense_754/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 dense_754/BiasAdd/ReadVariableOpReadVariableOp)dense_754_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0
dense_754/BiasAddBiasAdddense_754/MatMul:product:0(dense_754/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
6batch_normalization_680/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_680/moments/meanMeandense_754/BiasAdd:output:0?batch_normalization_680/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(
,batch_normalization_680/moments/StopGradientStopGradient-batch_normalization_680/moments/mean:output:0*
T0*
_output_shapes

:fË
1batch_normalization_680/moments/SquaredDifferenceSquaredDifferencedense_754/BiasAdd:output:05batch_normalization_680/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
:batch_normalization_680/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_680/moments/varianceMean5batch_normalization_680/moments/SquaredDifference:z:0Cbatch_normalization_680/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(
'batch_normalization_680/moments/SqueezeSqueeze-batch_normalization_680/moments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 £
)batch_normalization_680/moments/Squeeze_1Squeeze1batch_normalization_680/moments/variance:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 r
-batch_normalization_680/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_680/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_680_assignmovingavg_readvariableop_resource*
_output_shapes
:f*
dtype0É
+batch_normalization_680/AssignMovingAvg/subSub>batch_normalization_680/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_680/moments/Squeeze:output:0*
T0*
_output_shapes
:fÀ
+batch_normalization_680/AssignMovingAvg/mulMul/batch_normalization_680/AssignMovingAvg/sub:z:06batch_normalization_680/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f
'batch_normalization_680/AssignMovingAvgAssignSubVariableOp?batch_normalization_680_assignmovingavg_readvariableop_resource/batch_normalization_680/AssignMovingAvg/mul:z:07^batch_normalization_680/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_680/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_680/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_680_assignmovingavg_1_readvariableop_resource*
_output_shapes
:f*
dtype0Ï
-batch_normalization_680/AssignMovingAvg_1/subSub@batch_normalization_680/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_680/moments/Squeeze_1:output:0*
T0*
_output_shapes
:fÆ
-batch_normalization_680/AssignMovingAvg_1/mulMul1batch_normalization_680/AssignMovingAvg_1/sub:z:08batch_normalization_680/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f
)batch_normalization_680/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_680_assignmovingavg_1_readvariableop_resource1batch_normalization_680/AssignMovingAvg_1/mul:z:09^batch_normalization_680/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_680/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_680/batchnorm/addAddV22batch_normalization_680/moments/Squeeze_1:output:00batch_normalization_680/batchnorm/add/y:output:0*
T0*
_output_shapes
:f
'batch_normalization_680/batchnorm/RsqrtRsqrt)batch_normalization_680/batchnorm/add:z:0*
T0*
_output_shapes
:f®
4batch_normalization_680/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_680_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0¼
%batch_normalization_680/batchnorm/mulMul+batch_normalization_680/batchnorm/Rsqrt:y:0<batch_normalization_680/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f§
'batch_normalization_680/batchnorm/mul_1Muldense_754/BiasAdd:output:0)batch_normalization_680/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf°
'batch_normalization_680/batchnorm/mul_2Mul0batch_normalization_680/moments/Squeeze:output:0)batch_normalization_680/batchnorm/mul:z:0*
T0*
_output_shapes
:f¦
0batch_normalization_680/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_680_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0¸
%batch_normalization_680/batchnorm/subSub8batch_normalization_680/batchnorm/ReadVariableOp:value:0+batch_normalization_680/batchnorm/mul_2:z:0*
T0*
_output_shapes
:fº
'batch_normalization_680/batchnorm/add_1AddV2+batch_normalization_680/batchnorm/mul_1:z:0)batch_normalization_680/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
leaky_re_lu_680/LeakyRelu	LeakyRelu+batch_normalization_680/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*
alpha%>
dense_755/MatMul/ReadVariableOpReadVariableOp(dense_755_matmul_readvariableop_resource*
_output_shapes

:f*
dtype0
dense_755/MatMulMatMul'leaky_re_lu_680/LeakyRelu:activations:0'dense_755/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_755/BiasAdd/ReadVariableOpReadVariableOp)dense_755_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_755/BiasAddBiasAdddense_755/MatMul:product:0(dense_755/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_755/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp(^batch_normalization_676/AssignMovingAvg7^batch_normalization_676/AssignMovingAvg/ReadVariableOp*^batch_normalization_676/AssignMovingAvg_19^batch_normalization_676/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_676/batchnorm/ReadVariableOp5^batch_normalization_676/batchnorm/mul/ReadVariableOp(^batch_normalization_677/AssignMovingAvg7^batch_normalization_677/AssignMovingAvg/ReadVariableOp*^batch_normalization_677/AssignMovingAvg_19^batch_normalization_677/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_677/batchnorm/ReadVariableOp5^batch_normalization_677/batchnorm/mul/ReadVariableOp(^batch_normalization_678/AssignMovingAvg7^batch_normalization_678/AssignMovingAvg/ReadVariableOp*^batch_normalization_678/AssignMovingAvg_19^batch_normalization_678/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_678/batchnorm/ReadVariableOp5^batch_normalization_678/batchnorm/mul/ReadVariableOp(^batch_normalization_679/AssignMovingAvg7^batch_normalization_679/AssignMovingAvg/ReadVariableOp*^batch_normalization_679/AssignMovingAvg_19^batch_normalization_679/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_679/batchnorm/ReadVariableOp5^batch_normalization_679/batchnorm/mul/ReadVariableOp(^batch_normalization_680/AssignMovingAvg7^batch_normalization_680/AssignMovingAvg/ReadVariableOp*^batch_normalization_680/AssignMovingAvg_19^batch_normalization_680/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_680/batchnorm/ReadVariableOp5^batch_normalization_680/batchnorm/mul/ReadVariableOp!^dense_750/BiasAdd/ReadVariableOp ^dense_750/MatMul/ReadVariableOp!^dense_751/BiasAdd/ReadVariableOp ^dense_751/MatMul/ReadVariableOp!^dense_752/BiasAdd/ReadVariableOp ^dense_752/MatMul/ReadVariableOp!^dense_753/BiasAdd/ReadVariableOp ^dense_753/MatMul/ReadVariableOp!^dense_754/BiasAdd/ReadVariableOp ^dense_754/MatMul/ReadVariableOp!^dense_755/BiasAdd/ReadVariableOp ^dense_755/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization_676/AssignMovingAvg'batch_normalization_676/AssignMovingAvg2p
6batch_normalization_676/AssignMovingAvg/ReadVariableOp6batch_normalization_676/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_676/AssignMovingAvg_1)batch_normalization_676/AssignMovingAvg_12t
8batch_normalization_676/AssignMovingAvg_1/ReadVariableOp8batch_normalization_676/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_676/batchnorm/ReadVariableOp0batch_normalization_676/batchnorm/ReadVariableOp2l
4batch_normalization_676/batchnorm/mul/ReadVariableOp4batch_normalization_676/batchnorm/mul/ReadVariableOp2R
'batch_normalization_677/AssignMovingAvg'batch_normalization_677/AssignMovingAvg2p
6batch_normalization_677/AssignMovingAvg/ReadVariableOp6batch_normalization_677/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_677/AssignMovingAvg_1)batch_normalization_677/AssignMovingAvg_12t
8batch_normalization_677/AssignMovingAvg_1/ReadVariableOp8batch_normalization_677/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_677/batchnorm/ReadVariableOp0batch_normalization_677/batchnorm/ReadVariableOp2l
4batch_normalization_677/batchnorm/mul/ReadVariableOp4batch_normalization_677/batchnorm/mul/ReadVariableOp2R
'batch_normalization_678/AssignMovingAvg'batch_normalization_678/AssignMovingAvg2p
6batch_normalization_678/AssignMovingAvg/ReadVariableOp6batch_normalization_678/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_678/AssignMovingAvg_1)batch_normalization_678/AssignMovingAvg_12t
8batch_normalization_678/AssignMovingAvg_1/ReadVariableOp8batch_normalization_678/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_678/batchnorm/ReadVariableOp0batch_normalization_678/batchnorm/ReadVariableOp2l
4batch_normalization_678/batchnorm/mul/ReadVariableOp4batch_normalization_678/batchnorm/mul/ReadVariableOp2R
'batch_normalization_679/AssignMovingAvg'batch_normalization_679/AssignMovingAvg2p
6batch_normalization_679/AssignMovingAvg/ReadVariableOp6batch_normalization_679/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_679/AssignMovingAvg_1)batch_normalization_679/AssignMovingAvg_12t
8batch_normalization_679/AssignMovingAvg_1/ReadVariableOp8batch_normalization_679/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_679/batchnorm/ReadVariableOp0batch_normalization_679/batchnorm/ReadVariableOp2l
4batch_normalization_679/batchnorm/mul/ReadVariableOp4batch_normalization_679/batchnorm/mul/ReadVariableOp2R
'batch_normalization_680/AssignMovingAvg'batch_normalization_680/AssignMovingAvg2p
6batch_normalization_680/AssignMovingAvg/ReadVariableOp6batch_normalization_680/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_680/AssignMovingAvg_1)batch_normalization_680/AssignMovingAvg_12t
8batch_normalization_680/AssignMovingAvg_1/ReadVariableOp8batch_normalization_680/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_680/batchnorm/ReadVariableOp0batch_normalization_680/batchnorm/ReadVariableOp2l
4batch_normalization_680/batchnorm/mul/ReadVariableOp4batch_normalization_680/batchnorm/mul/ReadVariableOp2D
 dense_750/BiasAdd/ReadVariableOp dense_750/BiasAdd/ReadVariableOp2B
dense_750/MatMul/ReadVariableOpdense_750/MatMul/ReadVariableOp2D
 dense_751/BiasAdd/ReadVariableOp dense_751/BiasAdd/ReadVariableOp2B
dense_751/MatMul/ReadVariableOpdense_751/MatMul/ReadVariableOp2D
 dense_752/BiasAdd/ReadVariableOp dense_752/BiasAdd/ReadVariableOp2B
dense_752/MatMul/ReadVariableOpdense_752/MatMul/ReadVariableOp2D
 dense_753/BiasAdd/ReadVariableOp dense_753/BiasAdd/ReadVariableOp2B
dense_753/MatMul/ReadVariableOpdense_753/MatMul/ReadVariableOp2D
 dense_754/BiasAdd/ReadVariableOp dense_754/BiasAdd/ReadVariableOp2B
dense_754/MatMul/ReadVariableOpdense_754/MatMul/ReadVariableOp2D
 dense_755/BiasAdd/ReadVariableOp dense_755/BiasAdd/ReadVariableOp2B
dense_755/MatMul/ReadVariableOpdense_755/MatMul/ReadVariableOp:O K
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
0__inference_leaky_re_lu_679_layer_call_fn_751860

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
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_679_layer_call_and_return_conditional_losses_750119`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿG:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_678_layer_call_fn_751751

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
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_678_layer_call_and_return_conditional_losses_750087`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿG:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
ìY
£
I__inference_sequential_74_layer_call_and_return_conditional_losses_750497

inputs
normalization_74_sub_y
normalization_74_sqrt_x"
dense_750_750416:.
dense_750_750418:.,
batch_normalization_676_750421:.,
batch_normalization_676_750423:.,
batch_normalization_676_750425:.,
batch_normalization_676_750427:."
dense_751_750431:..
dense_751_750433:.,
batch_normalization_677_750436:.,
batch_normalization_677_750438:.,
batch_normalization_677_750440:.,
batch_normalization_677_750442:."
dense_752_750446:.G
dense_752_750448:G,
batch_normalization_678_750451:G,
batch_normalization_678_750453:G,
batch_normalization_678_750455:G,
batch_normalization_678_750457:G"
dense_753_750461:GG
dense_753_750463:G,
batch_normalization_679_750466:G,
batch_normalization_679_750468:G,
batch_normalization_679_750470:G,
batch_normalization_679_750472:G"
dense_754_750476:Gf
dense_754_750478:f,
batch_normalization_680_750481:f,
batch_normalization_680_750483:f,
batch_normalization_680_750485:f,
batch_normalization_680_750487:f"
dense_755_750491:f
dense_755_750493:
identity¢/batch_normalization_676/StatefulPartitionedCall¢/batch_normalization_677/StatefulPartitionedCall¢/batch_normalization_678/StatefulPartitionedCall¢/batch_normalization_679/StatefulPartitionedCall¢/batch_normalization_680/StatefulPartitionedCall¢!dense_750/StatefulPartitionedCall¢!dense_751/StatefulPartitionedCall¢!dense_752/StatefulPartitionedCall¢!dense_753/StatefulPartitionedCall¢!dense_754/StatefulPartitionedCall¢!dense_755/StatefulPartitionedCallm
normalization_74/subSubinputsnormalization_74_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_74/SqrtSqrtnormalization_74_sqrt_x*
T0*
_output_shapes

:_
normalization_74/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_74/MaximumMaximumnormalization_74/Sqrt:y:0#normalization_74/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_74/truedivRealDivnormalization_74/sub:z:0normalization_74/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_750/StatefulPartitionedCallStatefulPartitionedCallnormalization_74/truediv:z:0dense_750_750416dense_750_750418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_750_layer_call_and_return_conditional_losses_750003
/batch_normalization_676/StatefulPartitionedCallStatefulPartitionedCall*dense_750/StatefulPartitionedCall:output:0batch_normalization_676_750421batch_normalization_676_750423batch_normalization_676_750425batch_normalization_676_750427*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_676_layer_call_and_return_conditional_losses_749640ø
leaky_re_lu_676/PartitionedCallPartitionedCall8batch_normalization_676/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_676_layer_call_and_return_conditional_losses_750023
!dense_751/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_676/PartitionedCall:output:0dense_751_750431dense_751_750433*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_751_layer_call_and_return_conditional_losses_750035
/batch_normalization_677/StatefulPartitionedCallStatefulPartitionedCall*dense_751/StatefulPartitionedCall:output:0batch_normalization_677_750436batch_normalization_677_750438batch_normalization_677_750440batch_normalization_677_750442*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_677_layer_call_and_return_conditional_losses_749722ø
leaky_re_lu_677/PartitionedCallPartitionedCall8batch_normalization_677/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_677_layer_call_and_return_conditional_losses_750055
!dense_752/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_677/PartitionedCall:output:0dense_752_750446dense_752_750448*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_752_layer_call_and_return_conditional_losses_750067
/batch_normalization_678/StatefulPartitionedCallStatefulPartitionedCall*dense_752/StatefulPartitionedCall:output:0batch_normalization_678_750451batch_normalization_678_750453batch_normalization_678_750455batch_normalization_678_750457*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_678_layer_call_and_return_conditional_losses_749804ø
leaky_re_lu_678/PartitionedCallPartitionedCall8batch_normalization_678/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_678_layer_call_and_return_conditional_losses_750087
!dense_753/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_678/PartitionedCall:output:0dense_753_750461dense_753_750463*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_753_layer_call_and_return_conditional_losses_750099
/batch_normalization_679/StatefulPartitionedCallStatefulPartitionedCall*dense_753/StatefulPartitionedCall:output:0batch_normalization_679_750466batch_normalization_679_750468batch_normalization_679_750470batch_normalization_679_750472*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_679_layer_call_and_return_conditional_losses_749886ø
leaky_re_lu_679/PartitionedCallPartitionedCall8batch_normalization_679/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_679_layer_call_and_return_conditional_losses_750119
!dense_754/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_679/PartitionedCall:output:0dense_754_750476dense_754_750478*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_754_layer_call_and_return_conditional_losses_750131
/batch_normalization_680/StatefulPartitionedCallStatefulPartitionedCall*dense_754/StatefulPartitionedCall:output:0batch_normalization_680_750481batch_normalization_680_750483batch_normalization_680_750485batch_normalization_680_750487*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_680_layer_call_and_return_conditional_losses_749968ø
leaky_re_lu_680/PartitionedCallPartitionedCall8batch_normalization_680/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_680_layer_call_and_return_conditional_losses_750151
!dense_755/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_680/PartitionedCall:output:0dense_755_750491dense_755_750493*
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
E__inference_dense_755_layer_call_and_return_conditional_losses_750163y
IdentityIdentity*dense_755/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_676/StatefulPartitionedCall0^batch_normalization_677/StatefulPartitionedCall0^batch_normalization_678/StatefulPartitionedCall0^batch_normalization_679/StatefulPartitionedCall0^batch_normalization_680/StatefulPartitionedCall"^dense_750/StatefulPartitionedCall"^dense_751/StatefulPartitionedCall"^dense_752/StatefulPartitionedCall"^dense_753/StatefulPartitionedCall"^dense_754/StatefulPartitionedCall"^dense_755/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_676/StatefulPartitionedCall/batch_normalization_676/StatefulPartitionedCall2b
/batch_normalization_677/StatefulPartitionedCall/batch_normalization_677/StatefulPartitionedCall2b
/batch_normalization_678/StatefulPartitionedCall/batch_normalization_678/StatefulPartitionedCall2b
/batch_normalization_679/StatefulPartitionedCall/batch_normalization_679/StatefulPartitionedCall2b
/batch_normalization_680/StatefulPartitionedCall/batch_normalization_680/StatefulPartitionedCall2F
!dense_750/StatefulPartitionedCall!dense_750/StatefulPartitionedCall2F
!dense_751/StatefulPartitionedCall!dense_751/StatefulPartitionedCall2F
!dense_752/StatefulPartitionedCall!dense_752/StatefulPartitionedCall2F
!dense_753/StatefulPartitionedCall!dense_753/StatefulPartitionedCall2F
!dense_754/StatefulPartitionedCall!dense_754/StatefulPartitionedCall2F
!dense_755/StatefulPartitionedCall!dense_755/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ä

*__inference_dense_755_layer_call_fn_751983

inputs
unknown:f
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
E__inference_dense_755_layer_call_and_return_conditional_losses_750163o
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
:ÿÿÿÿÿÿÿÿÿf: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_678_layer_call_and_return_conditional_losses_749757

inputs/
!batchnorm_readvariableop_resource:G3
%batchnorm_mul_readvariableop_resource:G1
#batchnorm_readvariableop_1_resource:G1
#batchnorm_readvariableop_2_resource:G
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:G*
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
:GP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:G~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:G*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Gc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:G*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Gz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:G*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Gr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_677_layer_call_and_return_conditional_losses_749722

inputs5
'assignmovingavg_readvariableop_resource:.7
)assignmovingavg_1_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:./
!batchnorm_readvariableop_resource:.
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
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

:.
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
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
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:.*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:.x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:.¬
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
:.*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:.~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:.´
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
:ÿÿÿÿÿÿÿÿÿ.h
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
:ÿÿÿÿÿÿÿÿÿ.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_680_layer_call_fn_751969

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
:ÿÿÿÿÿÿÿÿÿf* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_680_layer_call_and_return_conditional_losses_750151`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿf:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs

â
.__inference_sequential_74_layer_call_fn_750241
normalization_74_input
unknown
	unknown_0
	unknown_1:.
	unknown_2:.
	unknown_3:.
	unknown_4:.
	unknown_5:.
	unknown_6:.
	unknown_7:..
	unknown_8:.
	unknown_9:.

unknown_10:.

unknown_11:.

unknown_12:.

unknown_13:.G

unknown_14:G

unknown_15:G

unknown_16:G

unknown_17:G

unknown_18:G

unknown_19:GG

unknown_20:G

unknown_21:G

unknown_22:G

unknown_23:G

unknown_24:G

unknown_25:Gf

unknown_26:f

unknown_27:f

unknown_28:f

unknown_29:f

unknown_30:f

unknown_31:f

unknown_32:
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallnormalization_74_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 !"*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_74_layer_call_and_return_conditional_losses_750170o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_74_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ð
²
S__inference_batch_normalization_676_layer_call_and_return_conditional_losses_749593

inputs/
!batchnorm_readvariableop_resource:.3
%batchnorm_mul_readvariableop_resource:.1
#batchnorm_readvariableop_1_resource:.1
#batchnorm_readvariableop_2_resource:.
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:.*
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
:ÿÿÿÿÿÿÿÿÿ.z
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
:ÿÿÿÿÿÿÿÿÿ.b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
î'
Ò
__inference_adapt_step_751429
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
ª
Ó
8__inference_batch_normalization_679_layer_call_fn_751801

inputs
unknown:G
	unknown_0:G
	unknown_1:G
	unknown_2:G
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_679_layer_call_and_return_conditional_losses_749886o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
È	
ö
E__inference_dense_752_layer_call_and_return_conditional_losses_750067

inputs0
matmul_readvariableop_resource:.G-
biasadd_readvariableop_resource:G
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:.G*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:G*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
Ä

*__inference_dense_751_layer_call_fn_751547

inputs
unknown:..
	unknown_0:.
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_751_layer_call_and_return_conditional_losses_750035o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
È	
ö
E__inference_dense_755_layer_call_and_return_conditional_losses_750163

inputs0
matmul_readvariableop_resource:f-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:f*
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
:ÿÿÿÿÿÿÿÿÿf: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
Ä

*__inference_dense_750_layer_call_fn_751438

inputs
unknown:.
	unknown_0:.
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_750_layer_call_and_return_conditional_losses_750003o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.`
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
 
_user_specified_nameinputs
È	
ö
E__inference_dense_755_layer_call_and_return_conditional_losses_751993

inputs0
matmul_readvariableop_resource:f-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:f*
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
:ÿÿÿÿÿÿÿÿÿf: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_678_layer_call_and_return_conditional_losses_751756

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿG:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
È	
ö
E__inference_dense_753_layer_call_and_return_conditional_losses_751775

inputs0
matmul_readvariableop_resource:GG-
biasadd_readvariableop_resource:G
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:GG*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:G*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿGw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_680_layer_call_and_return_conditional_losses_749968

inputs5
'assignmovingavg_readvariableop_resource:f7
)assignmovingavg_1_readvariableop_resource:f3
%batchnorm_mul_readvariableop_resource:f/
!batchnorm_readvariableop_resource:f
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:f
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:f*
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
:f*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:fx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f¬
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
:f*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:f~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f´
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
:fP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:f~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:fc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:fv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:fr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿfê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿf: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_679_layer_call_fn_751788

inputs
unknown:G
	unknown_0:G
	unknown_1:G
	unknown_2:G
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_679_layer_call_and_return_conditional_losses_749839o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
Ê
Ò
.__inference_sequential_74_layer_call_fn_750973

inputs
unknown
	unknown_0
	unknown_1:.
	unknown_2:.
	unknown_3:.
	unknown_4:.
	unknown_5:.
	unknown_6:.
	unknown_7:..
	unknown_8:.
	unknown_9:.

unknown_10:.

unknown_11:.

unknown_12:.

unknown_13:.G

unknown_14:G

unknown_15:G

unknown_16:G

unknown_17:G

unknown_18:G

unknown_19:GG

unknown_20:G

unknown_21:G

unknown_22:G

unknown_23:G

unknown_24:G

unknown_25:Gf

unknown_26:f

unknown_27:f

unknown_28:f

unknown_29:f

unknown_30:f

unknown_31:f

unknown_32:
identity¢StatefulPartitionedCall
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
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
 !"*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_74_layer_call_and_return_conditional_losses_750497o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ê
serving_default¶
Y
normalization_74_input?
(serving_default_normalization_74_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_7550
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:®®

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
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ó

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
 variance
 adapt_variance
	!count
"	keras_api
#_adapt_function"
_tf_keras_layer
»

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
,axis
	-gamma
.beta
/moving_mean
0moving_variance
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
»

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
^axis
	_gamma
`beta
amoving_mean
bmoving_variance
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
»

okernel
pbias
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
ì
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
¡kernel
	¢bias
£	variables
¤trainable_variables
¥regularization_losses
¦	keras_api
§__call__
+¨&call_and_return_all_conditional_losses"
_tf_keras_layer

	©iter
ªbeta_1
«beta_2

¬decay$m%m-m.m=m>mFmGmVmWm_m`mompmxmym	m	m	m	m	¡m	¢m$v%v-v .v¡=v¢>v£Fv¤Gv¥Vv¦Wv§_v¨`v©ovªpv«xv¬yv­	v®	v¯	v°	v±	¡v²	¢v³"
	optimizer
¶
0
 1
!2
$3
%4
-5
.6
/7
08
=9
>10
F11
G12
H13
I14
V15
W16
_17
`18
a19
b20
o21
p22
x23
y24
z25
{26
27
28
29
30
31
32
¡33
¢34"
trackable_list_wrapper
Ì
$0
%1
-2
.3
=4
>5
F6
G7
V8
W9
_10
`11
o12
p13
x14
y15
16
17
18
19
¡20
¢21"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_sequential_74_layer_call_fn_750241
.__inference_sequential_74_layer_call_fn_750900
.__inference_sequential_74_layer_call_fn_750973
.__inference_sequential_74_layer_call_fn_750641À
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
I__inference_sequential_74_layer_call_and_return_conditional_losses_751105
I__inference_sequential_74_layer_call_and_return_conditional_losses_751307
I__inference_sequential_74_layer_call_and_return_conditional_losses_750732
I__inference_sequential_74_layer_call_and_return_conditional_losses_750823À
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
!__inference__wrapped_model_749569normalization_74_input"
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
²serving_default"
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
__inference_adapt_step_751429
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
": .2dense_750/kernel
:.2dense_750/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_750_layer_call_fn_751438¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
E__inference_dense_750_layer_call_and_return_conditional_losses_751448¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:).2batch_normalization_676/gamma
*:(.2batch_normalization_676/beta
3:1. (2#batch_normalization_676/moving_mean
7:5. (2'batch_normalization_676/moving_variance
<
-0
.1
/2
03"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_676_layer_call_fn_751461
8__inference_batch_normalization_676_layer_call_fn_751474´
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
S__inference_batch_normalization_676_layer_call_and_return_conditional_losses_751494
S__inference_batch_normalization_676_layer_call_and_return_conditional_losses_751528´
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
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_676_layer_call_fn_751533¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
K__inference_leaky_re_lu_676_layer_call_and_return_conditional_losses_751538¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": ..2dense_751/kernel
:.2dense_751/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_751_layer_call_fn_751547¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
E__inference_dense_751_layer_call_and_return_conditional_losses_751557¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:).2batch_normalization_677/gamma
*:(.2batch_normalization_677/beta
3:1. (2#batch_normalization_677/moving_mean
7:5. (2'batch_normalization_677/moving_variance
<
F0
G1
H2
I3"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_677_layer_call_fn_751570
8__inference_batch_normalization_677_layer_call_fn_751583´
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
S__inference_batch_normalization_677_layer_call_and_return_conditional_losses_751603
S__inference_batch_normalization_677_layer_call_and_return_conditional_losses_751637´
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
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_677_layer_call_fn_751642¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
K__inference_leaky_re_lu_677_layer_call_and_return_conditional_losses_751647¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": .G2dense_752/kernel
:G2dense_752/bias
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_752_layer_call_fn_751656¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
E__inference_dense_752_layer_call_and_return_conditional_losses_751666¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)G2batch_normalization_678/gamma
*:(G2batch_normalization_678/beta
3:1G (2#batch_normalization_678/moving_mean
7:5G (2'batch_normalization_678/moving_variance
<
_0
`1
a2
b3"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_678_layer_call_fn_751679
8__inference_batch_normalization_678_layer_call_fn_751692´
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
S__inference_batch_normalization_678_layer_call_and_return_conditional_losses_751712
S__inference_batch_normalization_678_layer_call_and_return_conditional_losses_751746´
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
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_678_layer_call_fn_751751¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
K__inference_leaky_re_lu_678_layer_call_and_return_conditional_losses_751756¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": GG2dense_753/kernel
:G2dense_753/bias
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_753_layer_call_fn_751765¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
E__inference_dense_753_layer_call_and_return_conditional_losses_751775¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)G2batch_normalization_679/gamma
*:(G2batch_normalization_679/beta
3:1G (2#batch_normalization_679/moving_mean
7:5G (2'batch_normalization_679/moving_variance
<
x0
y1
z2
{3"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_679_layer_call_fn_751788
8__inference_batch_normalization_679_layer_call_fn_751801´
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
S__inference_batch_normalization_679_layer_call_and_return_conditional_losses_751821
S__inference_batch_normalization_679_layer_call_and_return_conditional_losses_751855´
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
ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_679_layer_call_fn_751860¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
K__inference_leaky_re_lu_679_layer_call_and_return_conditional_losses_751865¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": Gf2dense_754/kernel
:f2dense_754/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_754_layer_call_fn_751874¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
E__inference_dense_754_layer_call_and_return_conditional_losses_751884¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)f2batch_normalization_680/gamma
*:(f2batch_normalization_680/beta
3:1f (2#batch_normalization_680/moving_mean
7:5f (2'batch_normalization_680/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_680_layer_call_fn_751897
8__inference_batch_normalization_680_layer_call_fn_751910´
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
S__inference_batch_normalization_680_layer_call_and_return_conditional_losses_751930
S__inference_batch_normalization_680_layer_call_and_return_conditional_losses_751964´
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
ùnon_trainable_variables
úlayers
ûmetrics
 ülayer_regularization_losses
ýlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_680_layer_call_fn_751969¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
K__inference_leaky_re_lu_680_layer_call_and_return_conditional_losses_751974¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": f2dense_755/kernel
:2dense_755/bias
0
¡0
¢1"
trackable_list_wrapper
0
¡0
¢1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
£	variables
¤trainable_variables
¥regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_755_layer_call_fn_751983¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
E__inference_dense_755_layer_call_and_return_conditional_losses_751993¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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

0
 1
!2
/3
04
H5
I6
a7
b8
z9
{10
11
12"
trackable_list_wrapper

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
16"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÚB×
$__inference_signature_wrapper_751382normalization_74_input"
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
/0
01"
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
H0
I1"
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
a0
b1"
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
z0
{1"
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
0
1"
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

total

count
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
':%.2Adam/dense_750/kernel/m
!:.2Adam/dense_750/bias/m
0:..2$Adam/batch_normalization_676/gamma/m
/:-.2#Adam/batch_normalization_676/beta/m
':%..2Adam/dense_751/kernel/m
!:.2Adam/dense_751/bias/m
0:..2$Adam/batch_normalization_677/gamma/m
/:-.2#Adam/batch_normalization_677/beta/m
':%.G2Adam/dense_752/kernel/m
!:G2Adam/dense_752/bias/m
0:.G2$Adam/batch_normalization_678/gamma/m
/:-G2#Adam/batch_normalization_678/beta/m
':%GG2Adam/dense_753/kernel/m
!:G2Adam/dense_753/bias/m
0:.G2$Adam/batch_normalization_679/gamma/m
/:-G2#Adam/batch_normalization_679/beta/m
':%Gf2Adam/dense_754/kernel/m
!:f2Adam/dense_754/bias/m
0:.f2$Adam/batch_normalization_680/gamma/m
/:-f2#Adam/batch_normalization_680/beta/m
':%f2Adam/dense_755/kernel/m
!:2Adam/dense_755/bias/m
':%.2Adam/dense_750/kernel/v
!:.2Adam/dense_750/bias/v
0:..2$Adam/batch_normalization_676/gamma/v
/:-.2#Adam/batch_normalization_676/beta/v
':%..2Adam/dense_751/kernel/v
!:.2Adam/dense_751/bias/v
0:..2$Adam/batch_normalization_677/gamma/v
/:-.2#Adam/batch_normalization_677/beta/v
':%.G2Adam/dense_752/kernel/v
!:G2Adam/dense_752/bias/v
0:.G2$Adam/batch_normalization_678/gamma/v
/:-G2#Adam/batch_normalization_678/beta/v
':%GG2Adam/dense_753/kernel/v
!:G2Adam/dense_753/bias/v
0:.G2$Adam/batch_normalization_679/gamma/v
/:-G2#Adam/batch_normalization_679/beta/v
':%Gf2Adam/dense_754/kernel/v
!:f2Adam/dense_754/bias/v
0:.f2$Adam/batch_normalization_680/gamma/v
/:-f2#Adam/batch_normalization_680/beta/v
':%f2Adam/dense_755/kernel/v
!:2Adam/dense_755/bias/v
	J
Const
J	
Const_1Ì
!__inference__wrapped_model_749569¦,´µ$%0-/.=>IFHGVWb_a`op{xzy¡¢?¢<
5¢2
0-
normalization_74_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_755# 
	dense_755ÿÿÿÿÿÿÿÿÿo
__inference_adapt_step_751429N! C¢@
9¢6
41¢
ÿÿÿÿÿÿÿÿÿ	IteratorSpec 
ª "
 ¹
S__inference_batch_normalization_676_layer_call_and_return_conditional_losses_751494b0-/.3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ.
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ.
 ¹
S__inference_batch_normalization_676_layer_call_and_return_conditional_losses_751528b/0-.3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ.
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ.
 
8__inference_batch_normalization_676_layer_call_fn_751461U0-/.3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ.
p 
ª "ÿÿÿÿÿÿÿÿÿ.
8__inference_batch_normalization_676_layer_call_fn_751474U/0-.3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ.
p
ª "ÿÿÿÿÿÿÿÿÿ.¹
S__inference_batch_normalization_677_layer_call_and_return_conditional_losses_751603bIFHG3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ.
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ.
 ¹
S__inference_batch_normalization_677_layer_call_and_return_conditional_losses_751637bHIFG3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ.
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ.
 
8__inference_batch_normalization_677_layer_call_fn_751570UIFHG3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ.
p 
ª "ÿÿÿÿÿÿÿÿÿ.
8__inference_batch_normalization_677_layer_call_fn_751583UHIFG3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ.
p
ª "ÿÿÿÿÿÿÿÿÿ.¹
S__inference_batch_normalization_678_layer_call_and_return_conditional_losses_751712bb_a`3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿG
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿG
 ¹
S__inference_batch_normalization_678_layer_call_and_return_conditional_losses_751746bab_`3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿG
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿG
 
8__inference_batch_normalization_678_layer_call_fn_751679Ub_a`3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿG
p 
ª "ÿÿÿÿÿÿÿÿÿG
8__inference_batch_normalization_678_layer_call_fn_751692Uab_`3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿG
p
ª "ÿÿÿÿÿÿÿÿÿG¹
S__inference_batch_normalization_679_layer_call_and_return_conditional_losses_751821b{xzy3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿG
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿG
 ¹
S__inference_batch_normalization_679_layer_call_and_return_conditional_losses_751855bz{xy3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿG
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿG
 
8__inference_batch_normalization_679_layer_call_fn_751788U{xzy3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿG
p 
ª "ÿÿÿÿÿÿÿÿÿG
8__inference_batch_normalization_679_layer_call_fn_751801Uz{xy3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿG
p
ª "ÿÿÿÿÿÿÿÿÿG½
S__inference_batch_normalization_680_layer_call_and_return_conditional_losses_751930f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿf
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿf
 ½
S__inference_batch_normalization_680_layer_call_and_return_conditional_losses_751964f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿf
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿf
 
8__inference_batch_normalization_680_layer_call_fn_751897Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿf
p 
ª "ÿÿÿÿÿÿÿÿÿf
8__inference_batch_normalization_680_layer_call_fn_751910Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿf
p
ª "ÿÿÿÿÿÿÿÿÿf¥
E__inference_dense_750_layer_call_and_return_conditional_losses_751448\$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ.
 }
*__inference_dense_750_layer_call_fn_751438O$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ.¥
E__inference_dense_751_layer_call_and_return_conditional_losses_751557\=>/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ.
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ.
 }
*__inference_dense_751_layer_call_fn_751547O=>/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ.
ª "ÿÿÿÿÿÿÿÿÿ.¥
E__inference_dense_752_layer_call_and_return_conditional_losses_751666\VW/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ.
ª "%¢"

0ÿÿÿÿÿÿÿÿÿG
 }
*__inference_dense_752_layer_call_fn_751656OVW/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ.
ª "ÿÿÿÿÿÿÿÿÿG¥
E__inference_dense_753_layer_call_and_return_conditional_losses_751775\op/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿG
ª "%¢"

0ÿÿÿÿÿÿÿÿÿG
 }
*__inference_dense_753_layer_call_fn_751765Oop/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿG
ª "ÿÿÿÿÿÿÿÿÿG§
E__inference_dense_754_layer_call_and_return_conditional_losses_751884^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿG
ª "%¢"

0ÿÿÿÿÿÿÿÿÿf
 
*__inference_dense_754_layer_call_fn_751874Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿG
ª "ÿÿÿÿÿÿÿÿÿf§
E__inference_dense_755_layer_call_and_return_conditional_losses_751993^¡¢/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿf
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_755_layer_call_fn_751983Q¡¢/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿf
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_676_layer_call_and_return_conditional_losses_751538X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ.
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ.
 
0__inference_leaky_re_lu_676_layer_call_fn_751533K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ.
ª "ÿÿÿÿÿÿÿÿÿ.§
K__inference_leaky_re_lu_677_layer_call_and_return_conditional_losses_751647X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ.
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ.
 
0__inference_leaky_re_lu_677_layer_call_fn_751642K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ.
ª "ÿÿÿÿÿÿÿÿÿ.§
K__inference_leaky_re_lu_678_layer_call_and_return_conditional_losses_751756X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿG
ª "%¢"

0ÿÿÿÿÿÿÿÿÿG
 
0__inference_leaky_re_lu_678_layer_call_fn_751751K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿG
ª "ÿÿÿÿÿÿÿÿÿG§
K__inference_leaky_re_lu_679_layer_call_and_return_conditional_losses_751865X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿG
ª "%¢"

0ÿÿÿÿÿÿÿÿÿG
 
0__inference_leaky_re_lu_679_layer_call_fn_751860K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿG
ª "ÿÿÿÿÿÿÿÿÿG§
K__inference_leaky_re_lu_680_layer_call_and_return_conditional_losses_751974X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿf
ª "%¢"

0ÿÿÿÿÿÿÿÿÿf
 
0__inference_leaky_re_lu_680_layer_call_fn_751969K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿf
ª "ÿÿÿÿÿÿÿÿÿfì
I__inference_sequential_74_layer_call_and_return_conditional_losses_750732,´µ$%0-/.=>IFHGVWb_a`op{xzy¡¢G¢D
=¢:
0-
normalization_74_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ì
I__inference_sequential_74_layer_call_and_return_conditional_losses_750823,´µ$%/0-.=>HIFGVWab_`opz{xy¡¢G¢D
=¢:
0-
normalization_74_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ü
I__inference_sequential_74_layer_call_and_return_conditional_losses_751105,´µ$%0-/.=>IFHGVWb_a`op{xzy¡¢7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ü
I__inference_sequential_74_layer_call_and_return_conditional_losses_751307,´µ$%/0-.=>HIFGVWab_`opz{xy¡¢7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_sequential_74_layer_call_fn_750241,´µ$%0-/.=>IFHGVWb_a`op{xzy¡¢G¢D
=¢:
0-
normalization_74_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÄ
.__inference_sequential_74_layer_call_fn_750641,´µ$%/0-.=>HIFGVWab_`opz{xy¡¢G¢D
=¢:
0-
normalization_74_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ´
.__inference_sequential_74_layer_call_fn_750900,´µ$%0-/.=>IFHGVWb_a`op{xzy¡¢7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ´
.__inference_sequential_74_layer_call_fn_750973,´µ$%/0-.=>HIFGVWab_`opz{xy¡¢7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿé
$__inference_signature_wrapper_751382À,´µ$%0-/.=>IFHGVWb_a`op{xzy¡¢Y¢V
¢ 
OªL
J
normalization_74_input0-
normalization_74_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_755# 
	dense_755ÿÿÿÿÿÿÿÿÿ