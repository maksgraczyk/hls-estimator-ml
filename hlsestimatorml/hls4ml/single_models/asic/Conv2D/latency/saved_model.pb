ò¢'
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Êí#
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
z
dense_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7* 
shared_namedense_55/kernel
s
#dense_55/kernel/Read/ReadVariableOpReadVariableOpdense_55/kernel*
_output_shapes

:7*
dtype0
r
dense_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*
shared_namedense_55/bias
k
!dense_55/bias/Read/ReadVariableOpReadVariableOpdense_55/bias*
_output_shapes
:7*
dtype0

batch_normalization_49/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*-
shared_namebatch_normalization_49/gamma

0batch_normalization_49/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_49/gamma*
_output_shapes
:7*
dtype0

batch_normalization_49/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*,
shared_namebatch_normalization_49/beta

/batch_normalization_49/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_49/beta*
_output_shapes
:7*
dtype0

"batch_normalization_49/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*3
shared_name$"batch_normalization_49/moving_mean

6batch_normalization_49/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_49/moving_mean*
_output_shapes
:7*
dtype0
¤
&batch_normalization_49/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*7
shared_name(&batch_normalization_49/moving_variance

:batch_normalization_49/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_49/moving_variance*
_output_shapes
:7*
dtype0
z
dense_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:77* 
shared_namedense_56/kernel
s
#dense_56/kernel/Read/ReadVariableOpReadVariableOpdense_56/kernel*
_output_shapes

:77*
dtype0
r
dense_56/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*
shared_namedense_56/bias
k
!dense_56/bias/Read/ReadVariableOpReadVariableOpdense_56/bias*
_output_shapes
:7*
dtype0

batch_normalization_50/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*-
shared_namebatch_normalization_50/gamma

0batch_normalization_50/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_50/gamma*
_output_shapes
:7*
dtype0

batch_normalization_50/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*,
shared_namebatch_normalization_50/beta

/batch_normalization_50/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_50/beta*
_output_shapes
:7*
dtype0

"batch_normalization_50/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*3
shared_name$"batch_normalization_50/moving_mean

6batch_normalization_50/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_50/moving_mean*
_output_shapes
:7*
dtype0
¤
&batch_normalization_50/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*7
shared_name(&batch_normalization_50/moving_variance

:batch_normalization_50/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_50/moving_variance*
_output_shapes
:7*
dtype0
z
dense_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7u* 
shared_namedense_57/kernel
s
#dense_57/kernel/Read/ReadVariableOpReadVariableOpdense_57/kernel*
_output_shapes

:7u*
dtype0
r
dense_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:u*
shared_namedense_57/bias
k
!dense_57/bias/Read/ReadVariableOpReadVariableOpdense_57/bias*
_output_shapes
:u*
dtype0

batch_normalization_51/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:u*-
shared_namebatch_normalization_51/gamma

0batch_normalization_51/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_51/gamma*
_output_shapes
:u*
dtype0

batch_normalization_51/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:u*,
shared_namebatch_normalization_51/beta

/batch_normalization_51/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_51/beta*
_output_shapes
:u*
dtype0

"batch_normalization_51/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:u*3
shared_name$"batch_normalization_51/moving_mean

6batch_normalization_51/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_51/moving_mean*
_output_shapes
:u*
dtype0
¤
&batch_normalization_51/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:u*7
shared_name(&batch_normalization_51/moving_variance

:batch_normalization_51/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_51/moving_variance*
_output_shapes
:u*
dtype0
z
dense_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:u* 
shared_namedense_58/kernel
s
#dense_58/kernel/Read/ReadVariableOpReadVariableOpdense_58/kernel*
_output_shapes

:u*
dtype0
r
dense_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_58/bias
k
!dense_58/bias/Read/ReadVariableOpReadVariableOpdense_58/bias*
_output_shapes
:*
dtype0

batch_normalization_52/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_52/gamma

0batch_normalization_52/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_52/gamma*
_output_shapes
:*
dtype0

batch_normalization_52/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_52/beta

/batch_normalization_52/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_52/beta*
_output_shapes
:*
dtype0

"batch_normalization_52/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_52/moving_mean

6batch_normalization_52/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_52/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_52/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_52/moving_variance

:batch_normalization_52/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_52/moving_variance*
_output_shapes
:*
dtype0
z
dense_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_59/kernel
s
#dense_59/kernel/Read/ReadVariableOpReadVariableOpdense_59/kernel*
_output_shapes

:*
dtype0
r
dense_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_59/bias
k
!dense_59/bias/Read/ReadVariableOpReadVariableOpdense_59/bias*
_output_shapes
:*
dtype0

batch_normalization_53/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_53/gamma

0batch_normalization_53/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_53/gamma*
_output_shapes
:*
dtype0

batch_normalization_53/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_53/beta

/batch_normalization_53/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_53/beta*
_output_shapes
:*
dtype0

"batch_normalization_53/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_53/moving_mean

6batch_normalization_53/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_53/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_53/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_53/moving_variance

:batch_normalization_53/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_53/moving_variance*
_output_shapes
:*
dtype0
z
dense_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_60/kernel
s
#dense_60/kernel/Read/ReadVariableOpReadVariableOpdense_60/kernel*
_output_shapes

:*
dtype0
r
dense_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_60/bias
k
!dense_60/bias/Read/ReadVariableOpReadVariableOpdense_60/bias*
_output_shapes
:*
dtype0

batch_normalization_54/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_54/gamma

0batch_normalization_54/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_54/gamma*
_output_shapes
:*
dtype0

batch_normalization_54/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_54/beta

/batch_normalization_54/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_54/beta*
_output_shapes
:*
dtype0

"batch_normalization_54/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_54/moving_mean

6batch_normalization_54/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_54/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_54/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_54/moving_variance

:batch_normalization_54/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_54/moving_variance*
_output_shapes
:*
dtype0
z
dense_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_61/kernel
s
#dense_61/kernel/Read/ReadVariableOpReadVariableOpdense_61/kernel*
_output_shapes

:*
dtype0
r
dense_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_61/bias
k
!dense_61/bias/Read/ReadVariableOpReadVariableOpdense_61/bias*
_output_shapes
:*
dtype0

batch_normalization_55/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_55/gamma

0batch_normalization_55/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_55/gamma*
_output_shapes
:*
dtype0

batch_normalization_55/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_55/beta

/batch_normalization_55/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_55/beta*
_output_shapes
:*
dtype0

"batch_normalization_55/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_55/moving_mean

6batch_normalization_55/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_55/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_55/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_55/moving_variance

:batch_normalization_55/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_55/moving_variance*
_output_shapes
:*
dtype0
z
dense_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_62/kernel
s
#dense_62/kernel/Read/ReadVariableOpReadVariableOpdense_62/kernel*
_output_shapes

:*
dtype0
r
dense_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_62/bias
k
!dense_62/bias/Read/ReadVariableOpReadVariableOpdense_62/bias*
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

Adam/dense_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7*'
shared_nameAdam/dense_55/kernel/m

*Adam/dense_55/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_55/kernel/m*
_output_shapes

:7*
dtype0

Adam/dense_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*%
shared_nameAdam/dense_55/bias/m
y
(Adam/dense_55/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_55/bias/m*
_output_shapes
:7*
dtype0

#Adam/batch_normalization_49/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*4
shared_name%#Adam/batch_normalization_49/gamma/m

7Adam/batch_normalization_49/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_49/gamma/m*
_output_shapes
:7*
dtype0

"Adam/batch_normalization_49/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*3
shared_name$"Adam/batch_normalization_49/beta/m

6Adam/batch_normalization_49/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_49/beta/m*
_output_shapes
:7*
dtype0

Adam/dense_56/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:77*'
shared_nameAdam/dense_56/kernel/m

*Adam/dense_56/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/m*
_output_shapes

:77*
dtype0

Adam/dense_56/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*%
shared_nameAdam/dense_56/bias/m
y
(Adam/dense_56/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/m*
_output_shapes
:7*
dtype0

#Adam/batch_normalization_50/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*4
shared_name%#Adam/batch_normalization_50/gamma/m

7Adam/batch_normalization_50/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_50/gamma/m*
_output_shapes
:7*
dtype0

"Adam/batch_normalization_50/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*3
shared_name$"Adam/batch_normalization_50/beta/m

6Adam/batch_normalization_50/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_50/beta/m*
_output_shapes
:7*
dtype0

Adam/dense_57/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7u*'
shared_nameAdam/dense_57/kernel/m

*Adam/dense_57/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/m*
_output_shapes

:7u*
dtype0

Adam/dense_57/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:u*%
shared_nameAdam/dense_57/bias/m
y
(Adam/dense_57/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/m*
_output_shapes
:u*
dtype0

#Adam/batch_normalization_51/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:u*4
shared_name%#Adam/batch_normalization_51/gamma/m

7Adam/batch_normalization_51/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_51/gamma/m*
_output_shapes
:u*
dtype0

"Adam/batch_normalization_51/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:u*3
shared_name$"Adam/batch_normalization_51/beta/m

6Adam/batch_normalization_51/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_51/beta/m*
_output_shapes
:u*
dtype0

Adam/dense_58/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:u*'
shared_nameAdam/dense_58/kernel/m

*Adam/dense_58/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/m*
_output_shapes

:u*
dtype0

Adam/dense_58/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_58/bias/m
y
(Adam/dense_58/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_52/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_52/gamma/m

7Adam/batch_normalization_52/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_52/gamma/m*
_output_shapes
:*
dtype0

"Adam/batch_normalization_52/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_52/beta/m

6Adam/batch_normalization_52/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_52/beta/m*
_output_shapes
:*
dtype0

Adam/dense_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_59/kernel/m

*Adam/dense_59/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_59/bias/m
y
(Adam/dense_59/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_53/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_53/gamma/m

7Adam/batch_normalization_53/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_53/gamma/m*
_output_shapes
:*
dtype0

"Adam/batch_normalization_53/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_53/beta/m

6Adam/batch_normalization_53/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_53/beta/m*
_output_shapes
:*
dtype0

Adam/dense_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_60/kernel/m

*Adam/dense_60/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_60/bias/m
y
(Adam/dense_60/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_54/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_54/gamma/m

7Adam/batch_normalization_54/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_54/gamma/m*
_output_shapes
:*
dtype0

"Adam/batch_normalization_54/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_54/beta/m

6Adam/batch_normalization_54/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_54/beta/m*
_output_shapes
:*
dtype0

Adam/dense_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_61/kernel/m

*Adam/dense_61/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_61/bias/m
y
(Adam/dense_61/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_55/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_55/gamma/m

7Adam/batch_normalization_55/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_55/gamma/m*
_output_shapes
:*
dtype0

"Adam/batch_normalization_55/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_55/beta/m

6Adam/batch_normalization_55/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_55/beta/m*
_output_shapes
:*
dtype0

Adam/dense_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_62/kernel/m

*Adam/dense_62/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_62/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_62/bias/m
y
(Adam/dense_62/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_62/bias/m*
_output_shapes
:*
dtype0

Adam/dense_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7*'
shared_nameAdam/dense_55/kernel/v

*Adam/dense_55/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_55/kernel/v*
_output_shapes

:7*
dtype0

Adam/dense_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*%
shared_nameAdam/dense_55/bias/v
y
(Adam/dense_55/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_55/bias/v*
_output_shapes
:7*
dtype0

#Adam/batch_normalization_49/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*4
shared_name%#Adam/batch_normalization_49/gamma/v

7Adam/batch_normalization_49/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_49/gamma/v*
_output_shapes
:7*
dtype0

"Adam/batch_normalization_49/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*3
shared_name$"Adam/batch_normalization_49/beta/v

6Adam/batch_normalization_49/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_49/beta/v*
_output_shapes
:7*
dtype0

Adam/dense_56/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:77*'
shared_nameAdam/dense_56/kernel/v

*Adam/dense_56/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/v*
_output_shapes

:77*
dtype0

Adam/dense_56/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*%
shared_nameAdam/dense_56/bias/v
y
(Adam/dense_56/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/v*
_output_shapes
:7*
dtype0

#Adam/batch_normalization_50/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*4
shared_name%#Adam/batch_normalization_50/gamma/v

7Adam/batch_normalization_50/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_50/gamma/v*
_output_shapes
:7*
dtype0

"Adam/batch_normalization_50/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*3
shared_name$"Adam/batch_normalization_50/beta/v

6Adam/batch_normalization_50/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_50/beta/v*
_output_shapes
:7*
dtype0

Adam/dense_57/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7u*'
shared_nameAdam/dense_57/kernel/v

*Adam/dense_57/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/v*
_output_shapes

:7u*
dtype0

Adam/dense_57/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:u*%
shared_nameAdam/dense_57/bias/v
y
(Adam/dense_57/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/v*
_output_shapes
:u*
dtype0

#Adam/batch_normalization_51/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:u*4
shared_name%#Adam/batch_normalization_51/gamma/v

7Adam/batch_normalization_51/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_51/gamma/v*
_output_shapes
:u*
dtype0

"Adam/batch_normalization_51/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:u*3
shared_name$"Adam/batch_normalization_51/beta/v

6Adam/batch_normalization_51/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_51/beta/v*
_output_shapes
:u*
dtype0

Adam/dense_58/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:u*'
shared_nameAdam/dense_58/kernel/v

*Adam/dense_58/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/v*
_output_shapes

:u*
dtype0

Adam/dense_58/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_58/bias/v
y
(Adam/dense_58/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_52/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_52/gamma/v

7Adam/batch_normalization_52/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_52/gamma/v*
_output_shapes
:*
dtype0

"Adam/batch_normalization_52/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_52/beta/v

6Adam/batch_normalization_52/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_52/beta/v*
_output_shapes
:*
dtype0

Adam/dense_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_59/kernel/v

*Adam/dense_59/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_59/bias/v
y
(Adam/dense_59/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_53/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_53/gamma/v

7Adam/batch_normalization_53/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_53/gamma/v*
_output_shapes
:*
dtype0

"Adam/batch_normalization_53/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_53/beta/v

6Adam/batch_normalization_53/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_53/beta/v*
_output_shapes
:*
dtype0

Adam/dense_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_60/kernel/v

*Adam/dense_60/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_60/bias/v
y
(Adam/dense_60/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_60/bias/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_54/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_54/gamma/v

7Adam/batch_normalization_54/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_54/gamma/v*
_output_shapes
:*
dtype0

"Adam/batch_normalization_54/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_54/beta/v

6Adam/batch_normalization_54/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_54/beta/v*
_output_shapes
:*
dtype0

Adam/dense_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_61/kernel/v

*Adam/dense_61/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_61/bias/v
y
(Adam/dense_61/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_61/bias/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_55/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_55/gamma/v

7Adam/batch_normalization_55/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_55/gamma/v*
_output_shapes
:*
dtype0

"Adam/batch_normalization_55/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_55/beta/v

6Adam/batch_normalization_55/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_55/beta/v*
_output_shapes
:*
dtype0

Adam/dense_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_62/kernel/v

*Adam/dense_62/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_62/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_62/bias/v
y
(Adam/dense_62/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_62/bias/v*
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
üä
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*´ä
value©äB¥ä Bä
ª
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
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
 
signatures*
¾
!
_keep_axis
"_reduce_axis
#_reduce_axis_mask
$_broadcast_shape
%mean
%
adapt_mean
&variance
&adapt_variance
	'count
(	keras_api
)_adapt_function*
¦

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses*
Õ
2axis
	3gamma
4beta
5moving_mean
6moving_variance
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses*

=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
¦

Ckernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses*
Õ
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses*

V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses* 
¦

\kernel
]bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses*
Õ
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses*

o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses* 
¦

ukernel
vbias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses*
Ý
}axis
	~gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses*

¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses* 
®
§kernel
	¨bias
©	variables
ªtrainable_variables
«regularization_losses
¬	keras_api
­__call__
+®&call_and_return_all_conditional_losses*
à
	¯axis

°gamma
	±beta
²moving_mean
³moving_variance
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses*

º	variables
»trainable_variables
¼regularization_losses
½	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses* 
®
Àkernel
	Ábias
Â	variables
Ãtrainable_variables
Äregularization_losses
Å	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses*
à
	Èaxis

Égamma
	Êbeta
Ëmoving_mean
Ìmoving_variance
Í	variables
Îtrainable_variables
Ïregularization_losses
Ð	keras_api
Ñ__call__
+Ò&call_and_return_all_conditional_losses*

Ó	variables
Ôtrainable_variables
Õregularization_losses
Ö	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses* 
®
Ùkernel
	Úbias
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses*
©
	áiter
âbeta_1
ãbeta_2

ädecay*må+mæ3mç4mèCméDmêLmëMmì\mí]mîemïfmðumñvmò~mómô	mõ	mö	m÷	mø	§mù	¨mú	°mû	±mü	Àmý	Ámþ	Émÿ	Êm	Ùm	Úm*v+v3v4vCvDvLvMv\v]vevfvuvvv~vv	v	v	v	v	§v	¨v	°v	±v	Àv	Áv	Év	Êv	Ùv	Úv *

%0
&1
'2
*3
+4
35
46
57
68
C9
D10
L11
M12
N13
O14
\15
]16
e17
f18
g19
h20
u21
v22
~23
24
25
26
27
28
29
30
31
32
§33
¨34
°35
±36
²37
³38
À39
Á40
É41
Ê42
Ë43
Ì44
Ù45
Ú46*
ø
*0
+1
32
43
C4
D5
L6
M7
\8
]9
e10
f11
u12
v13
~14
15
16
17
18
19
§20
¨21
°22
±23
À24
Á25
É26
Ê27
Ù28
Ú29*
:
å0
æ1
ç2
è3
é4
ê5
ë6* 
µ
ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

ñserving_default* 
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
_Y
VARIABLE_VALUEdense_55/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_55/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

*0
+1*

*0
+1*


å0* 

ònon_trainable_variables
ólayers
ômetrics
 õlayer_regularization_losses
ölayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_49/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_49/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_49/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_49/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
30
41
52
63*

30
41*
* 

÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_56/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_56/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

C0
D1*

C0
D1*


æ0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_50/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_50/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_50/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_50/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
L0
M1
N2
O3*

L0
M1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_57/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_57/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

\0
]1*

\0
]1*


ç0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_51/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_51/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_51/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_51/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
e0
f1
g2
h3*

e0
f1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_58/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_58/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

u0
v1*

u0
v1*


è0* 

non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_52/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_52/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_52/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_52/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
"
~0
1
2
3*

~0
1*
* 

¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_59/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_59/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*


é0* 

®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_53/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_53/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_53/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_53/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_60/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_60/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

§0
¨1*

§0
¨1*


ê0* 

½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
©	variables
ªtrainable_variables
«regularization_losses
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_54/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_54/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_54/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_54/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
°0
±1
²2
³3*

°0
±1*
* 

Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
º	variables
»trainable_variables
¼regularization_losses
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_61/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_61/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

À0
Á1*

À0
Á1*


ë0* 

Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
Â	variables
Ãtrainable_variables
Äregularization_losses
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_55/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_55/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_55/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_55/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
É0
Ê1
Ë2
Ì3*

É0
Ê1*
* 

Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
Í	variables
Îtrainable_variables
Ïregularization_losses
Ñ__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
Ó	variables
Ôtrainable_variables
Õregularization_losses
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_62/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_62/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ù0
Ú1*

Ù0
Ú1*
* 

Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses*
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

%0
&1
'2
53
64
N5
O6
g7
h8
9
10
11
12
²13
³14
Ë15
Ì16*
²
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
22*

à0*
* 
* 
* 
* 
* 
* 


å0* 
* 

50
61*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


æ0* 
* 

N0
O1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


ç0* 
* 

g0
h1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


è0* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


é0* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


ê0* 
* 

²0
³1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


ë0* 
* 

Ë0
Ì1*
* 
* 
* 
* 
* 
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

átotal

âcount
ã	variables
ä	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

á0
â1*

ã	variables*
|
VARIABLE_VALUEAdam/dense_55/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_55/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_49/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_49/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_56/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_56/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_50/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_50/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_57/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_57/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_51/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_51/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_58/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_58/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_52/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_52/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_59/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_59/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_53/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_53/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_60/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_60/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_54/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_54/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_61/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_61/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_55/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_55/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_62/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_62/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_55/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_55/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_49/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_49/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_56/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_56/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_50/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_50/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_57/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_57/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_51/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_51/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_58/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_58/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_52/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_52/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_59/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_59/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_53/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_53/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_60/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_60/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_54/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_54/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_61/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_61/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_55/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_55/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_62/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_62/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

%serving_default_normalization_6_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Ü
StatefulPartitionedCallStatefulPartitionedCall%serving_default_normalization_6_inputConstConst_1dense_55/kerneldense_55/bias&batch_normalization_49/moving_variancebatch_normalization_49/gamma"batch_normalization_49/moving_meanbatch_normalization_49/betadense_56/kerneldense_56/bias&batch_normalization_50/moving_variancebatch_normalization_50/gamma"batch_normalization_50/moving_meanbatch_normalization_50/betadense_57/kerneldense_57/bias&batch_normalization_51/moving_variancebatch_normalization_51/gamma"batch_normalization_51/moving_meanbatch_normalization_51/betadense_58/kerneldense_58/bias&batch_normalization_52/moving_variancebatch_normalization_52/gamma"batch_normalization_52/moving_meanbatch_normalization_52/betadense_59/kerneldense_59/bias&batch_normalization_53/moving_variancebatch_normalization_53/gamma"batch_normalization_53/moving_meanbatch_normalization_53/betadense_60/kerneldense_60/bias&batch_normalization_54/moving_variancebatch_normalization_54/gamma"batch_normalization_54/moving_meanbatch_normalization_54/betadense_61/kerneldense_61/bias&batch_normalization_55/moving_variancebatch_normalization_55/gamma"batch_normalization_55/moving_meanbatch_normalization_55/betadense_62/kerneldense_62/bias*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1769583
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ê,
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp#dense_55/kernel/Read/ReadVariableOp!dense_55/bias/Read/ReadVariableOp0batch_normalization_49/gamma/Read/ReadVariableOp/batch_normalization_49/beta/Read/ReadVariableOp6batch_normalization_49/moving_mean/Read/ReadVariableOp:batch_normalization_49/moving_variance/Read/ReadVariableOp#dense_56/kernel/Read/ReadVariableOp!dense_56/bias/Read/ReadVariableOp0batch_normalization_50/gamma/Read/ReadVariableOp/batch_normalization_50/beta/Read/ReadVariableOp6batch_normalization_50/moving_mean/Read/ReadVariableOp:batch_normalization_50/moving_variance/Read/ReadVariableOp#dense_57/kernel/Read/ReadVariableOp!dense_57/bias/Read/ReadVariableOp0batch_normalization_51/gamma/Read/ReadVariableOp/batch_normalization_51/beta/Read/ReadVariableOp6batch_normalization_51/moving_mean/Read/ReadVariableOp:batch_normalization_51/moving_variance/Read/ReadVariableOp#dense_58/kernel/Read/ReadVariableOp!dense_58/bias/Read/ReadVariableOp0batch_normalization_52/gamma/Read/ReadVariableOp/batch_normalization_52/beta/Read/ReadVariableOp6batch_normalization_52/moving_mean/Read/ReadVariableOp:batch_normalization_52/moving_variance/Read/ReadVariableOp#dense_59/kernel/Read/ReadVariableOp!dense_59/bias/Read/ReadVariableOp0batch_normalization_53/gamma/Read/ReadVariableOp/batch_normalization_53/beta/Read/ReadVariableOp6batch_normalization_53/moving_mean/Read/ReadVariableOp:batch_normalization_53/moving_variance/Read/ReadVariableOp#dense_60/kernel/Read/ReadVariableOp!dense_60/bias/Read/ReadVariableOp0batch_normalization_54/gamma/Read/ReadVariableOp/batch_normalization_54/beta/Read/ReadVariableOp6batch_normalization_54/moving_mean/Read/ReadVariableOp:batch_normalization_54/moving_variance/Read/ReadVariableOp#dense_61/kernel/Read/ReadVariableOp!dense_61/bias/Read/ReadVariableOp0batch_normalization_55/gamma/Read/ReadVariableOp/batch_normalization_55/beta/Read/ReadVariableOp6batch_normalization_55/moving_mean/Read/ReadVariableOp:batch_normalization_55/moving_variance/Read/ReadVariableOp#dense_62/kernel/Read/ReadVariableOp!dense_62/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_55/kernel/m/Read/ReadVariableOp(Adam/dense_55/bias/m/Read/ReadVariableOp7Adam/batch_normalization_49/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_49/beta/m/Read/ReadVariableOp*Adam/dense_56/kernel/m/Read/ReadVariableOp(Adam/dense_56/bias/m/Read/ReadVariableOp7Adam/batch_normalization_50/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_50/beta/m/Read/ReadVariableOp*Adam/dense_57/kernel/m/Read/ReadVariableOp(Adam/dense_57/bias/m/Read/ReadVariableOp7Adam/batch_normalization_51/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_51/beta/m/Read/ReadVariableOp*Adam/dense_58/kernel/m/Read/ReadVariableOp(Adam/dense_58/bias/m/Read/ReadVariableOp7Adam/batch_normalization_52/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_52/beta/m/Read/ReadVariableOp*Adam/dense_59/kernel/m/Read/ReadVariableOp(Adam/dense_59/bias/m/Read/ReadVariableOp7Adam/batch_normalization_53/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_53/beta/m/Read/ReadVariableOp*Adam/dense_60/kernel/m/Read/ReadVariableOp(Adam/dense_60/bias/m/Read/ReadVariableOp7Adam/batch_normalization_54/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_54/beta/m/Read/ReadVariableOp*Adam/dense_61/kernel/m/Read/ReadVariableOp(Adam/dense_61/bias/m/Read/ReadVariableOp7Adam/batch_normalization_55/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_55/beta/m/Read/ReadVariableOp*Adam/dense_62/kernel/m/Read/ReadVariableOp(Adam/dense_62/bias/m/Read/ReadVariableOp*Adam/dense_55/kernel/v/Read/ReadVariableOp(Adam/dense_55/bias/v/Read/ReadVariableOp7Adam/batch_normalization_49/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_49/beta/v/Read/ReadVariableOp*Adam/dense_56/kernel/v/Read/ReadVariableOp(Adam/dense_56/bias/v/Read/ReadVariableOp7Adam/batch_normalization_50/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_50/beta/v/Read/ReadVariableOp*Adam/dense_57/kernel/v/Read/ReadVariableOp(Adam/dense_57/bias/v/Read/ReadVariableOp7Adam/batch_normalization_51/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_51/beta/v/Read/ReadVariableOp*Adam/dense_58/kernel/v/Read/ReadVariableOp(Adam/dense_58/bias/v/Read/ReadVariableOp7Adam/batch_normalization_52/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_52/beta/v/Read/ReadVariableOp*Adam/dense_59/kernel/v/Read/ReadVariableOp(Adam/dense_59/bias/v/Read/ReadVariableOp7Adam/batch_normalization_53/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_53/beta/v/Read/ReadVariableOp*Adam/dense_60/kernel/v/Read/ReadVariableOp(Adam/dense_60/bias/v/Read/ReadVariableOp7Adam/batch_normalization_54/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_54/beta/v/Read/ReadVariableOp*Adam/dense_61/kernel/v/Read/ReadVariableOp(Adam/dense_61/bias/v/Read/ReadVariableOp7Adam/batch_normalization_55/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_55/beta/v/Read/ReadVariableOp*Adam/dense_62/kernel/v/Read/ReadVariableOp(Adam/dense_62/bias/v/Read/ReadVariableOpConst_2*~
Tinw
u2s		*
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
 __inference__traced_save_1770937
ï
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_55/kerneldense_55/biasbatch_normalization_49/gammabatch_normalization_49/beta"batch_normalization_49/moving_mean&batch_normalization_49/moving_variancedense_56/kerneldense_56/biasbatch_normalization_50/gammabatch_normalization_50/beta"batch_normalization_50/moving_mean&batch_normalization_50/moving_variancedense_57/kerneldense_57/biasbatch_normalization_51/gammabatch_normalization_51/beta"batch_normalization_51/moving_mean&batch_normalization_51/moving_variancedense_58/kerneldense_58/biasbatch_normalization_52/gammabatch_normalization_52/beta"batch_normalization_52/moving_mean&batch_normalization_52/moving_variancedense_59/kerneldense_59/biasbatch_normalization_53/gammabatch_normalization_53/beta"batch_normalization_53/moving_mean&batch_normalization_53/moving_variancedense_60/kerneldense_60/biasbatch_normalization_54/gammabatch_normalization_54/beta"batch_normalization_54/moving_mean&batch_normalization_54/moving_variancedense_61/kerneldense_61/biasbatch_normalization_55/gammabatch_normalization_55/beta"batch_normalization_55/moving_mean&batch_normalization_55/moving_variancedense_62/kerneldense_62/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount_1Adam/dense_55/kernel/mAdam/dense_55/bias/m#Adam/batch_normalization_49/gamma/m"Adam/batch_normalization_49/beta/mAdam/dense_56/kernel/mAdam/dense_56/bias/m#Adam/batch_normalization_50/gamma/m"Adam/batch_normalization_50/beta/mAdam/dense_57/kernel/mAdam/dense_57/bias/m#Adam/batch_normalization_51/gamma/m"Adam/batch_normalization_51/beta/mAdam/dense_58/kernel/mAdam/dense_58/bias/m#Adam/batch_normalization_52/gamma/m"Adam/batch_normalization_52/beta/mAdam/dense_59/kernel/mAdam/dense_59/bias/m#Adam/batch_normalization_53/gamma/m"Adam/batch_normalization_53/beta/mAdam/dense_60/kernel/mAdam/dense_60/bias/m#Adam/batch_normalization_54/gamma/m"Adam/batch_normalization_54/beta/mAdam/dense_61/kernel/mAdam/dense_61/bias/m#Adam/batch_normalization_55/gamma/m"Adam/batch_normalization_55/beta/mAdam/dense_62/kernel/mAdam/dense_62/bias/mAdam/dense_55/kernel/vAdam/dense_55/bias/v#Adam/batch_normalization_49/gamma/v"Adam/batch_normalization_49/beta/vAdam/dense_56/kernel/vAdam/dense_56/bias/v#Adam/batch_normalization_50/gamma/v"Adam/batch_normalization_50/beta/vAdam/dense_57/kernel/vAdam/dense_57/bias/v#Adam/batch_normalization_51/gamma/v"Adam/batch_normalization_51/beta/vAdam/dense_58/kernel/vAdam/dense_58/bias/v#Adam/batch_normalization_52/gamma/v"Adam/batch_normalization_52/beta/vAdam/dense_59/kernel/vAdam/dense_59/bias/v#Adam/batch_normalization_53/gamma/v"Adam/batch_normalization_53/beta/vAdam/dense_60/kernel/vAdam/dense_60/bias/v#Adam/batch_normalization_54/gamma/v"Adam/batch_normalization_54/beta/vAdam/dense_61/kernel/vAdam/dense_61/bias/v#Adam/batch_normalization_55/gamma/v"Adam/batch_normalization_55/beta/vAdam/dense_62/kernel/vAdam/dense_62/bias/v*}
Tinv
t2r*
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
#__inference__traced_restore_1771286´½
%
ì
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1767031

inputs5
'assignmovingavg_readvariableop_resource:u7
)assignmovingavg_1_readvariableop_resource:u3
%batchnorm_mul_readvariableop_resource:u/
!batchnorm_readvariableop_resource:u
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:u*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:u
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿul
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:u*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:u*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:u*
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
:u*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ux
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:u¬
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
:u*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:u~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:u´
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
:uP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:u~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:u*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:uc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿuh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:uv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:u*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ur
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿub
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿuê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿu: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1770191

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_54_layer_call_fn_1770292

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1767277o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_1767610

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1770312

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1767066

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¬
__inference_loss_fn_2_1770529I
7dense_57_kernel_regularizer_abs_readvariableop_resource:7u
identity¢.dense_57/kernel/Regularizer/Abs/ReadVariableOp¦
.dense_57/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_57_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:7u*
dtype0
dense_57/kernel/Regularizer/AbsAbs6dense_57/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7ur
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum#dense_57/kernel/Regularizer/Abs:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *û9);
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_57/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^dense_57/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_57/kernel/Regularizer/Abs/ReadVariableOp.dense_57/kernel/Regularizer/Abs/ReadVariableOp

¬
__inference_loss_fn_5_1770562I
7dense_60_kernel_regularizer_abs_readvariableop_resource:
identity¢.dense_60/kernel/Regularizer/Abs/ReadVariableOp¦
.dense_60/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_60_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:*
dtype0
dense_60/kernel/Regularizer/AbsAbs6dense_60/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_60/kernel/Regularizer/SumSum#dense_60/kernel/Regularizer/Abs:y:0*dense_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_60/kernel/Regularizer/mulMul*dense_60/kernel/Regularizer/mul/x:output:0(dense_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_60/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^dense_60/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_60/kernel/Regularizer/Abs/ReadVariableOp.dense_60/kernel/Regularizer/Abs/ReadVariableOp
È	
ö
E__inference_dense_62_layer_call_and_return_conditional_losses_1767660

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1766949

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
Ó


%__inference_signature_wrapper_1769583
normalization_6_input
unknown
	unknown_0
	unknown_1:7
	unknown_2:7
	unknown_3:7
	unknown_4:7
	unknown_5:7
	unknown_6:7
	unknown_7:77
	unknown_8:7
	unknown_9:7

unknown_10:7

unknown_11:7

unknown_12:7

unknown_13:7u

unknown_14:u

unknown_15:u

unknown_16:u

unknown_17:u

unknown_18:u

unknown_19:u

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallnormalization_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1766796o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namenormalization_6_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ð
²
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1767148

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
§
E__inference_dense_61_layer_call_and_return_conditional_losses_1767628

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense_61/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.dense_61/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_61/kernel/Regularizer/AbsAbs6dense_61/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_61/kernel/Regularizer/SumSum#dense_61/kernel/Regularizer/Abs:y:0*dense_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_61/kernel/Regularizer/mulMul*dense_61/kernel/Regularizer/mul/x:output:0(dense_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_61/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_61/kernel/Regularizer/Abs/ReadVariableOp.dense_61/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1770225

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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
Ä

*__inference_dense_57_layer_call_fn_1769887

inputs
unknown:7u
	unknown_0:u
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_57_layer_call_and_return_conditional_losses_1767476o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu`
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
Ð
²
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1770070

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
°J
#__inference__traced_restore_1771286
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 4
"assignvariableop_3_dense_55_kernel:7.
 assignvariableop_4_dense_55_bias:7=
/assignvariableop_5_batch_normalization_49_gamma:7<
.assignvariableop_6_batch_normalization_49_beta:7C
5assignvariableop_7_batch_normalization_49_moving_mean:7G
9assignvariableop_8_batch_normalization_49_moving_variance:74
"assignvariableop_9_dense_56_kernel:77/
!assignvariableop_10_dense_56_bias:7>
0assignvariableop_11_batch_normalization_50_gamma:7=
/assignvariableop_12_batch_normalization_50_beta:7D
6assignvariableop_13_batch_normalization_50_moving_mean:7H
:assignvariableop_14_batch_normalization_50_moving_variance:75
#assignvariableop_15_dense_57_kernel:7u/
!assignvariableop_16_dense_57_bias:u>
0assignvariableop_17_batch_normalization_51_gamma:u=
/assignvariableop_18_batch_normalization_51_beta:uD
6assignvariableop_19_batch_normalization_51_moving_mean:uH
:assignvariableop_20_batch_normalization_51_moving_variance:u5
#assignvariableop_21_dense_58_kernel:u/
!assignvariableop_22_dense_58_bias:>
0assignvariableop_23_batch_normalization_52_gamma:=
/assignvariableop_24_batch_normalization_52_beta:D
6assignvariableop_25_batch_normalization_52_moving_mean:H
:assignvariableop_26_batch_normalization_52_moving_variance:5
#assignvariableop_27_dense_59_kernel:/
!assignvariableop_28_dense_59_bias:>
0assignvariableop_29_batch_normalization_53_gamma:=
/assignvariableop_30_batch_normalization_53_beta:D
6assignvariableop_31_batch_normalization_53_moving_mean:H
:assignvariableop_32_batch_normalization_53_moving_variance:5
#assignvariableop_33_dense_60_kernel:/
!assignvariableop_34_dense_60_bias:>
0assignvariableop_35_batch_normalization_54_gamma:=
/assignvariableop_36_batch_normalization_54_beta:D
6assignvariableop_37_batch_normalization_54_moving_mean:H
:assignvariableop_38_batch_normalization_54_moving_variance:5
#assignvariableop_39_dense_61_kernel:/
!assignvariableop_40_dense_61_bias:>
0assignvariableop_41_batch_normalization_55_gamma:=
/assignvariableop_42_batch_normalization_55_beta:D
6assignvariableop_43_batch_normalization_55_moving_mean:H
:assignvariableop_44_batch_normalization_55_moving_variance:5
#assignvariableop_45_dense_62_kernel:/
!assignvariableop_46_dense_62_bias:'
assignvariableop_47_adam_iter:	 )
assignvariableop_48_adam_beta_1: )
assignvariableop_49_adam_beta_2: (
assignvariableop_50_adam_decay: #
assignvariableop_51_total: %
assignvariableop_52_count_1: <
*assignvariableop_53_adam_dense_55_kernel_m:76
(assignvariableop_54_adam_dense_55_bias_m:7E
7assignvariableop_55_adam_batch_normalization_49_gamma_m:7D
6assignvariableop_56_adam_batch_normalization_49_beta_m:7<
*assignvariableop_57_adam_dense_56_kernel_m:776
(assignvariableop_58_adam_dense_56_bias_m:7E
7assignvariableop_59_adam_batch_normalization_50_gamma_m:7D
6assignvariableop_60_adam_batch_normalization_50_beta_m:7<
*assignvariableop_61_adam_dense_57_kernel_m:7u6
(assignvariableop_62_adam_dense_57_bias_m:uE
7assignvariableop_63_adam_batch_normalization_51_gamma_m:uD
6assignvariableop_64_adam_batch_normalization_51_beta_m:u<
*assignvariableop_65_adam_dense_58_kernel_m:u6
(assignvariableop_66_adam_dense_58_bias_m:E
7assignvariableop_67_adam_batch_normalization_52_gamma_m:D
6assignvariableop_68_adam_batch_normalization_52_beta_m:<
*assignvariableop_69_adam_dense_59_kernel_m:6
(assignvariableop_70_adam_dense_59_bias_m:E
7assignvariableop_71_adam_batch_normalization_53_gamma_m:D
6assignvariableop_72_adam_batch_normalization_53_beta_m:<
*assignvariableop_73_adam_dense_60_kernel_m:6
(assignvariableop_74_adam_dense_60_bias_m:E
7assignvariableop_75_adam_batch_normalization_54_gamma_m:D
6assignvariableop_76_adam_batch_normalization_54_beta_m:<
*assignvariableop_77_adam_dense_61_kernel_m:6
(assignvariableop_78_adam_dense_61_bias_m:E
7assignvariableop_79_adam_batch_normalization_55_gamma_m:D
6assignvariableop_80_adam_batch_normalization_55_beta_m:<
*assignvariableop_81_adam_dense_62_kernel_m:6
(assignvariableop_82_adam_dense_62_bias_m:<
*assignvariableop_83_adam_dense_55_kernel_v:76
(assignvariableop_84_adam_dense_55_bias_v:7E
7assignvariableop_85_adam_batch_normalization_49_gamma_v:7D
6assignvariableop_86_adam_batch_normalization_49_beta_v:7<
*assignvariableop_87_adam_dense_56_kernel_v:776
(assignvariableop_88_adam_dense_56_bias_v:7E
7assignvariableop_89_adam_batch_normalization_50_gamma_v:7D
6assignvariableop_90_adam_batch_normalization_50_beta_v:7<
*assignvariableop_91_adam_dense_57_kernel_v:7u6
(assignvariableop_92_adam_dense_57_bias_v:uE
7assignvariableop_93_adam_batch_normalization_51_gamma_v:uD
6assignvariableop_94_adam_batch_normalization_51_beta_v:u<
*assignvariableop_95_adam_dense_58_kernel_v:u6
(assignvariableop_96_adam_dense_58_bias_v:E
7assignvariableop_97_adam_batch_normalization_52_gamma_v:D
6assignvariableop_98_adam_batch_normalization_52_beta_v:<
*assignvariableop_99_adam_dense_59_kernel_v:7
)assignvariableop_100_adam_dense_59_bias_v:F
8assignvariableop_101_adam_batch_normalization_53_gamma_v:E
7assignvariableop_102_adam_batch_normalization_53_beta_v:=
+assignvariableop_103_adam_dense_60_kernel_v:7
)assignvariableop_104_adam_dense_60_bias_v:F
8assignvariableop_105_adam_batch_normalization_54_gamma_v:E
7assignvariableop_106_adam_batch_normalization_54_beta_v:=
+assignvariableop_107_adam_dense_61_kernel_v:7
)assignvariableop_108_adam_dense_61_bias_v:F
8assignvariableop_109_adam_batch_normalization_55_gamma_v:E
7assignvariableop_110_adam_batch_normalization_55_beta_v:=
+assignvariableop_111_adam_dense_62_kernel_v:7
)assignvariableop_112_adam_dense_62_bias_v:
identity_114¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_105¢AssignVariableOp_106¢AssignVariableOp_107¢AssignVariableOp_108¢AssignVariableOp_109¢AssignVariableOp_11¢AssignVariableOp_110¢AssignVariableOp_111¢AssignVariableOp_112¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_99¾?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:r*
dtype0*ä>
valueÚ>B×>rB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH×
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:r*
dtype0*ù
valueïBìrB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ü
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Þ
_output_shapesË
È::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypesv
t2r		[
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
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_55_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_55_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp/assignvariableop_5_batch_normalization_49_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_49_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_7AssignVariableOp5assignvariableop_7_batch_normalization_49_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_8AssignVariableOp9assignvariableop_8_batch_normalization_49_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_56_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense_56_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_11AssignVariableOp0assignvariableop_11_batch_normalization_50_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_50_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_13AssignVariableOp6assignvariableop_13_batch_normalization_50_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_14AssignVariableOp:assignvariableop_14_batch_normalization_50_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp#assignvariableop_15_dense_57_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp!assignvariableop_16_dense_57_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_17AssignVariableOp0assignvariableop_17_batch_normalization_51_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_51_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_19AssignVariableOp6assignvariableop_19_batch_normalization_51_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_20AssignVariableOp:assignvariableop_20_batch_normalization_51_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp#assignvariableop_21_dense_58_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp!assignvariableop_22_dense_58_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_23AssignVariableOp0assignvariableop_23_batch_normalization_52_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_24AssignVariableOp/assignvariableop_24_batch_normalization_52_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_25AssignVariableOp6assignvariableop_25_batch_normalization_52_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_26AssignVariableOp:assignvariableop_26_batch_normalization_52_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp#assignvariableop_27_dense_59_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp!assignvariableop_28_dense_59_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_29AssignVariableOp0assignvariableop_29_batch_normalization_53_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_30AssignVariableOp/assignvariableop_30_batch_normalization_53_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_31AssignVariableOp6assignvariableop_31_batch_normalization_53_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_32AssignVariableOp:assignvariableop_32_batch_normalization_53_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp#assignvariableop_33_dense_60_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp!assignvariableop_34_dense_60_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_35AssignVariableOp0assignvariableop_35_batch_normalization_54_gammaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_36AssignVariableOp/assignvariableop_36_batch_normalization_54_betaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_37AssignVariableOp6assignvariableop_37_batch_normalization_54_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_38AssignVariableOp:assignvariableop_38_batch_normalization_54_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp#assignvariableop_39_dense_61_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp!assignvariableop_40_dense_61_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_41AssignVariableOp0assignvariableop_41_batch_normalization_55_gammaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_42AssignVariableOp/assignvariableop_42_batch_normalization_55_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_43AssignVariableOp6assignvariableop_43_batch_normalization_55_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_44AssignVariableOp:assignvariableop_44_batch_normalization_55_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp#assignvariableop_45_dense_62_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp!assignvariableop_46_dense_62_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_47AssignVariableOpassignvariableop_47_adam_iterIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOpassignvariableop_48_adam_beta_1Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOpassignvariableop_49_adam_beta_2Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOpassignvariableop_50_adam_decayIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOpassignvariableop_51_totalIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOpassignvariableop_52_count_1Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_55_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_55_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_55AssignVariableOp7assignvariableop_55_adam_batch_normalization_49_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adam_batch_normalization_49_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_dense_56_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_dense_56_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_batch_normalization_50_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_50_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_57_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_57_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_batch_normalization_51_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_51_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_58_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_58_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_batch_normalization_52_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_52_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_dense_59_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_dense_59_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_71AssignVariableOp7assignvariableop_71_adam_batch_normalization_53_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_batch_normalization_53_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_dense_60_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_dense_60_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_75AssignVariableOp7assignvariableop_75_adam_batch_normalization_54_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adam_batch_normalization_54_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_dense_61_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_dense_61_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_batch_normalization_55_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_batch_normalization_55_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_dense_62_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_dense_62_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_dense_55_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adam_dense_55_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_85AssignVariableOp7assignvariableop_85_adam_batch_normalization_49_gamma_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_86AssignVariableOp6assignvariableop_86_adam_batch_normalization_49_beta_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_dense_56_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_dense_56_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_89AssignVariableOp7assignvariableop_89_adam_batch_normalization_50_gamma_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_90AssignVariableOp6assignvariableop_90_adam_batch_normalization_50_beta_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_dense_57_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_dense_57_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_93AssignVariableOp7assignvariableop_93_adam_batch_normalization_51_gamma_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_94AssignVariableOp6assignvariableop_94_adam_batch_normalization_51_beta_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_95AssignVariableOp*assignvariableop_95_adam_dense_58_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_96AssignVariableOp(assignvariableop_96_adam_dense_58_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_97AssignVariableOp7assignvariableop_97_adam_batch_normalization_52_gamma_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_98AssignVariableOp6assignvariableop_98_adam_batch_normalization_52_beta_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_99AssignVariableOp*assignvariableop_99_adam_dense_59_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_100AssignVariableOp)assignvariableop_100_adam_dense_59_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_101AssignVariableOp8assignvariableop_101_adam_batch_normalization_53_gamma_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_102AssignVariableOp7assignvariableop_102_adam_batch_normalization_53_beta_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_dense_60_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_dense_60_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_105AssignVariableOp8assignvariableop_105_adam_batch_normalization_54_gamma_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_106AssignVariableOp7assignvariableop_106_adam_batch_normalization_54_beta_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_107AssignVariableOp+assignvariableop_107_adam_dense_61_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_108AssignVariableOp)assignvariableop_108_adam_dense_61_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_109AssignVariableOp8assignvariableop_109_adam_batch_normalization_55_gamma_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_110AssignVariableOp7assignvariableop_110_adam_batch_normalization_55_beta_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_111AssignVariableOp+assignvariableop_111_adam_dense_62_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_112AssignVariableOp)assignvariableop_112_adam_dense_62_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_113Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_114IdentityIdentity_113:output:0^NoOp_1*
T0*
_output_shapes
: ÿ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_114Identity_114:output:0*ù
_input_shapesç
ä: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_112AssignVariableOp_1122*
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
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
å
g
K__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_1769751

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
Ä

*__inference_dense_55_layer_call_fn_1769645

inputs
unknown:7
	unknown_0:7
identity¢StatefulPartitionedCallÚ
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
GPU 2J 8 *N
fIRG
E__inference_dense_55_layer_call_and_return_conditional_losses_1767400o
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
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_1767496

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿu:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 
_user_specified_nameinputs
¾
§
E__inference_dense_55_layer_call_and_return_conditional_losses_1769661

inputs0
matmul_readvariableop_resource:7-
biasadd_readvariableop_resource:7
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense_55/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7*
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
:ÿÿÿÿÿÿÿÿÿ7
.dense_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7*
dtype0
dense_55/kernel/Regularizer/AbsAbs6dense_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7r
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_55/kernel/Regularizer/SumSum#dense_55/kernel/Regularizer/Abs:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¨
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_55/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_55/kernel/Regularizer/Abs/ReadVariableOp.dense_55/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_1767458

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
õ
¡

.__inference_sequential_6_layer_call_fn_1768380
normalization_6_input
unknown
	unknown_0
	unknown_1:7
	unknown_2:7
	unknown_3:7
	unknown_4:7
	unknown_5:7
	unknown_6:7
	unknown_7:77
	unknown_8:7
	unknown_9:7

unknown_10:7

unknown_11:7

unknown_12:7

unknown_13:7u

unknown_14:u

unknown_15:u

unknown_16:u

unknown_17:u

unknown_18:u

unknown_19:u

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallnormalization_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*@
_read_only_resource_inputs"
 	
 !"%&'(+,-.*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_1768188o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namenormalization_6_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ä

*__inference_dense_62_layer_call_fn_1770486

inputs
unknown:
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
E__inference_dense_62_layer_call_and_return_conditional_losses_1767660o
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
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1767359

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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
Ä

*__inference_dense_61_layer_call_fn_1770371

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_61_layer_call_and_return_conditional_losses_1767628o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_53_layer_call_fn_1770171

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1767195o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_1769993

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿu:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_1770235

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
§
E__inference_dense_60_layer_call_and_return_conditional_losses_1767590

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense_60/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.dense_60/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_60/kernel/Regularizer/AbsAbs6dense_60/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_60/kernel/Regularizer/SumSum#dense_60/kernel/Regularizer/Abs:y:0*dense_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_60/kernel/Regularizer/mulMul*dense_60/kernel/Regularizer/mul/x:output:0(dense_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_60/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_60/kernel/Regularizer/Abs/ReadVariableOp.dense_60/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô­

I__inference_sequential_6_layer_call_and_return_conditional_losses_1768188

inputs
normalization_6_sub_y
normalization_6_sqrt_x"
dense_55_1768035:7
dense_55_1768037:7,
batch_normalization_49_1768040:7,
batch_normalization_49_1768042:7,
batch_normalization_49_1768044:7,
batch_normalization_49_1768046:7"
dense_56_1768050:77
dense_56_1768052:7,
batch_normalization_50_1768055:7,
batch_normalization_50_1768057:7,
batch_normalization_50_1768059:7,
batch_normalization_50_1768061:7"
dense_57_1768065:7u
dense_57_1768067:u,
batch_normalization_51_1768070:u,
batch_normalization_51_1768072:u,
batch_normalization_51_1768074:u,
batch_normalization_51_1768076:u"
dense_58_1768080:u
dense_58_1768082:,
batch_normalization_52_1768085:,
batch_normalization_52_1768087:,
batch_normalization_52_1768089:,
batch_normalization_52_1768091:"
dense_59_1768095:
dense_59_1768097:,
batch_normalization_53_1768100:,
batch_normalization_53_1768102:,
batch_normalization_53_1768104:,
batch_normalization_53_1768106:"
dense_60_1768110:
dense_60_1768112:,
batch_normalization_54_1768115:,
batch_normalization_54_1768117:,
batch_normalization_54_1768119:,
batch_normalization_54_1768121:"
dense_61_1768125:
dense_61_1768127:,
batch_normalization_55_1768130:,
batch_normalization_55_1768132:,
batch_normalization_55_1768134:,
batch_normalization_55_1768136:"
dense_62_1768140:
dense_62_1768142:
identity¢.batch_normalization_49/StatefulPartitionedCall¢.batch_normalization_50/StatefulPartitionedCall¢.batch_normalization_51/StatefulPartitionedCall¢.batch_normalization_52/StatefulPartitionedCall¢.batch_normalization_53/StatefulPartitionedCall¢.batch_normalization_54/StatefulPartitionedCall¢.batch_normalization_55/StatefulPartitionedCall¢ dense_55/StatefulPartitionedCall¢.dense_55/kernel/Regularizer/Abs/ReadVariableOp¢ dense_56/StatefulPartitionedCall¢.dense_56/kernel/Regularizer/Abs/ReadVariableOp¢ dense_57/StatefulPartitionedCall¢.dense_57/kernel/Regularizer/Abs/ReadVariableOp¢ dense_58/StatefulPartitionedCall¢.dense_58/kernel/Regularizer/Abs/ReadVariableOp¢ dense_59/StatefulPartitionedCall¢.dense_59/kernel/Regularizer/Abs/ReadVariableOp¢ dense_60/StatefulPartitionedCall¢.dense_60/kernel/Regularizer/Abs/ReadVariableOp¢ dense_61/StatefulPartitionedCall¢.dense_61/kernel/Regularizer/Abs/ReadVariableOp¢ dense_62/StatefulPartitionedCallk
normalization_6/subSubinputsnormalization_6_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
normalization_6/SqrtSqrtnormalization_6_sqrt_x*
T0*
_output_shapes

:^
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_55/StatefulPartitionedCallStatefulPartitionedCallnormalization_6/truediv:z:0dense_55_1768035dense_55_1768037*
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
GPU 2J 8 *N
fIRG
E__inference_dense_55_layer_call_and_return_conditional_losses_1767400
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0batch_normalization_49_1768040batch_normalization_49_1768042batch_normalization_49_1768044batch_normalization_49_1768046*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1766867ö
leaky_re_lu_49/PartitionedCallPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_1767420
 dense_56/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_49/PartitionedCall:output:0dense_56_1768050dense_56_1768052*
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
GPU 2J 8 *N
fIRG
E__inference_dense_56_layer_call_and_return_conditional_losses_1767438
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0batch_normalization_50_1768055batch_normalization_50_1768057batch_normalization_50_1768059batch_normalization_50_1768061*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1766949ö
leaky_re_lu_50/PartitionedCallPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_1767458
 dense_57/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_50/PartitionedCall:output:0dense_57_1768065dense_57_1768067*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_57_layer_call_and_return_conditional_losses_1767476
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0batch_normalization_51_1768070batch_normalization_51_1768072batch_normalization_51_1768074batch_normalization_51_1768076*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1767031ö
leaky_re_lu_51/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_1767496
 dense_58/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_51/PartitionedCall:output:0dense_58_1768080dense_58_1768082*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_58_layer_call_and_return_conditional_losses_1767514
.batch_normalization_52/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0batch_normalization_52_1768085batch_normalization_52_1768087batch_normalization_52_1768089batch_normalization_52_1768091*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1767113ö
leaky_re_lu_52/PartitionedCallPartitionedCall7batch_normalization_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_1767534
 dense_59/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_52/PartitionedCall:output:0dense_59_1768095dense_59_1768097*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_1767552
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall)dense_59/StatefulPartitionedCall:output:0batch_normalization_53_1768100batch_normalization_53_1768102batch_normalization_53_1768104batch_normalization_53_1768106*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1767195ö
leaky_re_lu_53/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_1767572
 dense_60/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_53/PartitionedCall:output:0dense_60_1768110dense_60_1768112*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_60_layer_call_and_return_conditional_losses_1767590
.batch_normalization_54/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0batch_normalization_54_1768115batch_normalization_54_1768117batch_normalization_54_1768119batch_normalization_54_1768121*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1767277ö
leaky_re_lu_54/PartitionedCallPartitionedCall7batch_normalization_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_1767610
 dense_61/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_54/PartitionedCall:output:0dense_61_1768125dense_61_1768127*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_61_layer_call_and_return_conditional_losses_1767628
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0batch_normalization_55_1768130batch_normalization_55_1768132batch_normalization_55_1768134batch_normalization_55_1768136*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1767359ö
leaky_re_lu_55/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_1767648
 dense_62/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_55/PartitionedCall:output:0dense_62_1768140dense_62_1768142*
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
E__inference_dense_62_layer_call_and_return_conditional_losses_1767660
.dense_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_55_1768035*
_output_shapes

:7*
dtype0
dense_55/kernel/Regularizer/AbsAbs6dense_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7r
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_55/kernel/Regularizer/SumSum#dense_55/kernel/Regularizer/Abs:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_56/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_56_1768050*
_output_shapes

:77*
dtype0
dense_56/kernel/Regularizer/AbsAbs6dense_56/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77r
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum#dense_56/kernel/Regularizer/Abs:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_57/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_57_1768065*
_output_shapes

:7u*
dtype0
dense_57/kernel/Regularizer/AbsAbs6dense_57/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7ur
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum#dense_57/kernel/Regularizer/Abs:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *û9);
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_58/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_58_1768080*
_output_shapes

:u*
dtype0
dense_58/kernel/Regularizer/AbsAbs6dense_58/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:ur
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum#dense_58/kernel/Regularizer/Abs:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_59/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_59_1768095*
_output_shapes

:*
dtype0
dense_59/kernel/Regularizer/AbsAbs6dense_59/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum#dense_59/kernel/Regularizer/Abs:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_60/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_60_1768110*
_output_shapes

:*
dtype0
dense_60/kernel/Regularizer/AbsAbs6dense_60/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_60/kernel/Regularizer/SumSum#dense_60/kernel/Regularizer/Abs:y:0*dense_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_60/kernel/Regularizer/mulMul*dense_60/kernel/Regularizer/mul/x:output:0(dense_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_61/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_61_1768125*
_output_shapes

:*
dtype0
dense_61/kernel/Regularizer/AbsAbs6dense_61/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_61/kernel/Regularizer/SumSum#dense_61/kernel/Regularizer/Abs:y:0*dense_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_61/kernel/Regularizer/mulMul*dense_61/kernel/Regularizer/mul/x:output:0(dense_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_62/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_52/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_54/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall/^dense_55/kernel/Regularizer/Abs/ReadVariableOp!^dense_56/StatefulPartitionedCall/^dense_56/kernel/Regularizer/Abs/ReadVariableOp!^dense_57/StatefulPartitionedCall/^dense_57/kernel/Regularizer/Abs/ReadVariableOp!^dense_58/StatefulPartitionedCall/^dense_58/kernel/Regularizer/Abs/ReadVariableOp!^dense_59/StatefulPartitionedCall/^dense_59/kernel/Regularizer/Abs/ReadVariableOp!^dense_60/StatefulPartitionedCall/^dense_60/kernel/Regularizer/Abs/ReadVariableOp!^dense_61/StatefulPartitionedCall/^dense_61/kernel/Regularizer/Abs/ReadVariableOp!^dense_62/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_52/StatefulPartitionedCall.batch_normalization_52/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_54/StatefulPartitionedCall.batch_normalization_54/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2`
.dense_55/kernel/Regularizer/Abs/ReadVariableOp.dense_55/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2`
.dense_56/kernel/Regularizer/Abs/ReadVariableOp.dense_56/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2`
.dense_57/kernel/Regularizer/Abs/ReadVariableOp.dense_57/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2`
.dense_58/kernel/Regularizer/Abs/ReadVariableOp.dense_58/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2`
.dense_59/kernel/Regularizer/Abs/ReadVariableOp.dense_59/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2`
.dense_60/kernel/Regularizer/Abs/ReadVariableOp.dense_60/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2`
.dense_61/kernel/Regularizer/Abs/ReadVariableOp.dense_61/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
ª
Ó
8__inference_batch_normalization_55_layer_call_fn_1770413

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1767359o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1766984

inputs/
!batchnorm_readvariableop_resource:u3
%batchnorm_mul_readvariableop_resource:u1
#batchnorm_readvariableop_1_resource:u1
#batchnorm_readvariableop_2_resource:u
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:u*
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
:uP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:u~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:u*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:uc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿuz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:u*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:uz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:u*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ur
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿub
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿuº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿu: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 
_user_specified_nameinputs
²
î*
I__inference_sequential_6_layer_call_and_return_conditional_losses_1769166

inputs
normalization_6_sub_y
normalization_6_sqrt_x9
'dense_55_matmul_readvariableop_resource:76
(dense_55_biasadd_readvariableop_resource:7F
8batch_normalization_49_batchnorm_readvariableop_resource:7J
<batch_normalization_49_batchnorm_mul_readvariableop_resource:7H
:batch_normalization_49_batchnorm_readvariableop_1_resource:7H
:batch_normalization_49_batchnorm_readvariableop_2_resource:79
'dense_56_matmul_readvariableop_resource:776
(dense_56_biasadd_readvariableop_resource:7F
8batch_normalization_50_batchnorm_readvariableop_resource:7J
<batch_normalization_50_batchnorm_mul_readvariableop_resource:7H
:batch_normalization_50_batchnorm_readvariableop_1_resource:7H
:batch_normalization_50_batchnorm_readvariableop_2_resource:79
'dense_57_matmul_readvariableop_resource:7u6
(dense_57_biasadd_readvariableop_resource:uF
8batch_normalization_51_batchnorm_readvariableop_resource:uJ
<batch_normalization_51_batchnorm_mul_readvariableop_resource:uH
:batch_normalization_51_batchnorm_readvariableop_1_resource:uH
:batch_normalization_51_batchnorm_readvariableop_2_resource:u9
'dense_58_matmul_readvariableop_resource:u6
(dense_58_biasadd_readvariableop_resource:F
8batch_normalization_52_batchnorm_readvariableop_resource:J
<batch_normalization_52_batchnorm_mul_readvariableop_resource:H
:batch_normalization_52_batchnorm_readvariableop_1_resource:H
:batch_normalization_52_batchnorm_readvariableop_2_resource:9
'dense_59_matmul_readvariableop_resource:6
(dense_59_biasadd_readvariableop_resource:F
8batch_normalization_53_batchnorm_readvariableop_resource:J
<batch_normalization_53_batchnorm_mul_readvariableop_resource:H
:batch_normalization_53_batchnorm_readvariableop_1_resource:H
:batch_normalization_53_batchnorm_readvariableop_2_resource:9
'dense_60_matmul_readvariableop_resource:6
(dense_60_biasadd_readvariableop_resource:F
8batch_normalization_54_batchnorm_readvariableop_resource:J
<batch_normalization_54_batchnorm_mul_readvariableop_resource:H
:batch_normalization_54_batchnorm_readvariableop_1_resource:H
:batch_normalization_54_batchnorm_readvariableop_2_resource:9
'dense_61_matmul_readvariableop_resource:6
(dense_61_biasadd_readvariableop_resource:F
8batch_normalization_55_batchnorm_readvariableop_resource:J
<batch_normalization_55_batchnorm_mul_readvariableop_resource:H
:batch_normalization_55_batchnorm_readvariableop_1_resource:H
:batch_normalization_55_batchnorm_readvariableop_2_resource:9
'dense_62_matmul_readvariableop_resource:6
(dense_62_biasadd_readvariableop_resource:
identity¢/batch_normalization_49/batchnorm/ReadVariableOp¢1batch_normalization_49/batchnorm/ReadVariableOp_1¢1batch_normalization_49/batchnorm/ReadVariableOp_2¢3batch_normalization_49/batchnorm/mul/ReadVariableOp¢/batch_normalization_50/batchnorm/ReadVariableOp¢1batch_normalization_50/batchnorm/ReadVariableOp_1¢1batch_normalization_50/batchnorm/ReadVariableOp_2¢3batch_normalization_50/batchnorm/mul/ReadVariableOp¢/batch_normalization_51/batchnorm/ReadVariableOp¢1batch_normalization_51/batchnorm/ReadVariableOp_1¢1batch_normalization_51/batchnorm/ReadVariableOp_2¢3batch_normalization_51/batchnorm/mul/ReadVariableOp¢/batch_normalization_52/batchnorm/ReadVariableOp¢1batch_normalization_52/batchnorm/ReadVariableOp_1¢1batch_normalization_52/batchnorm/ReadVariableOp_2¢3batch_normalization_52/batchnorm/mul/ReadVariableOp¢/batch_normalization_53/batchnorm/ReadVariableOp¢1batch_normalization_53/batchnorm/ReadVariableOp_1¢1batch_normalization_53/batchnorm/ReadVariableOp_2¢3batch_normalization_53/batchnorm/mul/ReadVariableOp¢/batch_normalization_54/batchnorm/ReadVariableOp¢1batch_normalization_54/batchnorm/ReadVariableOp_1¢1batch_normalization_54/batchnorm/ReadVariableOp_2¢3batch_normalization_54/batchnorm/mul/ReadVariableOp¢/batch_normalization_55/batchnorm/ReadVariableOp¢1batch_normalization_55/batchnorm/ReadVariableOp_1¢1batch_normalization_55/batchnorm/ReadVariableOp_2¢3batch_normalization_55/batchnorm/mul/ReadVariableOp¢dense_55/BiasAdd/ReadVariableOp¢dense_55/MatMul/ReadVariableOp¢.dense_55/kernel/Regularizer/Abs/ReadVariableOp¢dense_56/BiasAdd/ReadVariableOp¢dense_56/MatMul/ReadVariableOp¢.dense_56/kernel/Regularizer/Abs/ReadVariableOp¢dense_57/BiasAdd/ReadVariableOp¢dense_57/MatMul/ReadVariableOp¢.dense_57/kernel/Regularizer/Abs/ReadVariableOp¢dense_58/BiasAdd/ReadVariableOp¢dense_58/MatMul/ReadVariableOp¢.dense_58/kernel/Regularizer/Abs/ReadVariableOp¢dense_59/BiasAdd/ReadVariableOp¢dense_59/MatMul/ReadVariableOp¢.dense_59/kernel/Regularizer/Abs/ReadVariableOp¢dense_60/BiasAdd/ReadVariableOp¢dense_60/MatMul/ReadVariableOp¢.dense_60/kernel/Regularizer/Abs/ReadVariableOp¢dense_61/BiasAdd/ReadVariableOp¢dense_61/MatMul/ReadVariableOp¢.dense_61/kernel/Regularizer/Abs/ReadVariableOp¢dense_62/BiasAdd/ReadVariableOp¢dense_62/MatMul/ReadVariableOpk
normalization_6/subSubinputsnormalization_6_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
normalization_6/SqrtSqrtnormalization_6_sqrt_x*
T0*
_output_shapes

:^
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

:7*
dtype0
dense_55/MatMulMatMulnormalization_6/truediv:z:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
dense_55/BiasAdd/ReadVariableOpReadVariableOp(dense_55_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¤
/batch_normalization_49/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_49_batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0k
&batch_normalization_49/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¼
$batch_normalization_49/batchnorm/addAddV27batch_normalization_49/batchnorm/ReadVariableOp:value:0/batch_normalization_49/batchnorm/add/y:output:0*
T0*
_output_shapes
:7~
&batch_normalization_49/batchnorm/RsqrtRsqrt(batch_normalization_49/batchnorm/add:z:0*
T0*
_output_shapes
:7¬
3batch_normalization_49/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_49_batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0¹
$batch_normalization_49/batchnorm/mulMul*batch_normalization_49/batchnorm/Rsqrt:y:0;batch_normalization_49/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7¤
&batch_normalization_49/batchnorm/mul_1Muldense_55/BiasAdd:output:0(batch_normalization_49/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¨
1batch_normalization_49/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_49_batchnorm_readvariableop_1_resource*
_output_shapes
:7*
dtype0·
&batch_normalization_49/batchnorm/mul_2Mul9batch_normalization_49/batchnorm/ReadVariableOp_1:value:0(batch_normalization_49/batchnorm/mul:z:0*
T0*
_output_shapes
:7¨
1batch_normalization_49/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_49_batchnorm_readvariableop_2_resource*
_output_shapes
:7*
dtype0·
$batch_normalization_49/batchnorm/subSub9batch_normalization_49/batchnorm/ReadVariableOp_2:value:0*batch_normalization_49/batchnorm/mul_2:z:0*
T0*
_output_shapes
:7·
&batch_normalization_49/batchnorm/add_1AddV2*batch_normalization_49/batchnorm/mul_1:z:0(batch_normalization_49/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
leaky_re_lu_49/LeakyRelu	LeakyRelu*batch_normalization_49/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource*
_output_shapes

:77*
dtype0
dense_56/MatMulMatMul&leaky_re_lu_49/LeakyRelu:activations:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
dense_56/BiasAdd/ReadVariableOpReadVariableOp(dense_56_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¤
/batch_normalization_50/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_50_batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0k
&batch_normalization_50/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¼
$batch_normalization_50/batchnorm/addAddV27batch_normalization_50/batchnorm/ReadVariableOp:value:0/batch_normalization_50/batchnorm/add/y:output:0*
T0*
_output_shapes
:7~
&batch_normalization_50/batchnorm/RsqrtRsqrt(batch_normalization_50/batchnorm/add:z:0*
T0*
_output_shapes
:7¬
3batch_normalization_50/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_50_batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0¹
$batch_normalization_50/batchnorm/mulMul*batch_normalization_50/batchnorm/Rsqrt:y:0;batch_normalization_50/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7¤
&batch_normalization_50/batchnorm/mul_1Muldense_56/BiasAdd:output:0(batch_normalization_50/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¨
1batch_normalization_50/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_50_batchnorm_readvariableop_1_resource*
_output_shapes
:7*
dtype0·
&batch_normalization_50/batchnorm/mul_2Mul9batch_normalization_50/batchnorm/ReadVariableOp_1:value:0(batch_normalization_50/batchnorm/mul:z:0*
T0*
_output_shapes
:7¨
1batch_normalization_50/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_50_batchnorm_readvariableop_2_resource*
_output_shapes
:7*
dtype0·
$batch_normalization_50/batchnorm/subSub9batch_normalization_50/batchnorm/ReadVariableOp_2:value:0*batch_normalization_50/batchnorm/mul_2:z:0*
T0*
_output_shapes
:7·
&batch_normalization_50/batchnorm/add_1AddV2*batch_normalization_50/batchnorm/mul_1:z:0(batch_normalization_50/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
leaky_re_lu_50/LeakyRelu	LeakyRelu*batch_normalization_50/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes

:7u*
dtype0
dense_57/MatMulMatMul&leaky_re_lu_50/LeakyRelu:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:u*
dtype0
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu¤
/batch_normalization_51/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_51_batchnorm_readvariableop_resource*
_output_shapes
:u*
dtype0k
&batch_normalization_51/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¼
$batch_normalization_51/batchnorm/addAddV27batch_normalization_51/batchnorm/ReadVariableOp:value:0/batch_normalization_51/batchnorm/add/y:output:0*
T0*
_output_shapes
:u~
&batch_normalization_51/batchnorm/RsqrtRsqrt(batch_normalization_51/batchnorm/add:z:0*
T0*
_output_shapes
:u¬
3batch_normalization_51/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_51_batchnorm_mul_readvariableop_resource*
_output_shapes
:u*
dtype0¹
$batch_normalization_51/batchnorm/mulMul*batch_normalization_51/batchnorm/Rsqrt:y:0;batch_normalization_51/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:u¤
&batch_normalization_51/batchnorm/mul_1Muldense_57/BiasAdd:output:0(batch_normalization_51/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu¨
1batch_normalization_51/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_51_batchnorm_readvariableop_1_resource*
_output_shapes
:u*
dtype0·
&batch_normalization_51/batchnorm/mul_2Mul9batch_normalization_51/batchnorm/ReadVariableOp_1:value:0(batch_normalization_51/batchnorm/mul:z:0*
T0*
_output_shapes
:u¨
1batch_normalization_51/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_51_batchnorm_readvariableop_2_resource*
_output_shapes
:u*
dtype0·
$batch_normalization_51/batchnorm/subSub9batch_normalization_51/batchnorm/ReadVariableOp_2:value:0*batch_normalization_51/batchnorm/mul_2:z:0*
T0*
_output_shapes
:u·
&batch_normalization_51/batchnorm/add_1AddV2*batch_normalization_51/batchnorm/mul_1:z:0(batch_normalization_51/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
leaky_re_lu_51/LeakyRelu	LeakyRelu*batch_normalization_51/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*
alpha%>
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

:u*
dtype0
dense_58/MatMulMatMul&leaky_re_lu_51/LeakyRelu:activations:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_58/BiasAdd/ReadVariableOpReadVariableOp(dense_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/batch_normalization_52/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_52_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_52/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¼
$batch_normalization_52/batchnorm/addAddV27batch_normalization_52/batchnorm/ReadVariableOp:value:0/batch_normalization_52/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_52/batchnorm/RsqrtRsqrt(batch_normalization_52/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_52/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_52_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_52/batchnorm/mulMul*batch_normalization_52/batchnorm/Rsqrt:y:0;batch_normalization_52/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¤
&batch_normalization_52/batchnorm/mul_1Muldense_58/BiasAdd:output:0(batch_normalization_52/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
1batch_normalization_52/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_52_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0·
&batch_normalization_52/batchnorm/mul_2Mul9batch_normalization_52/batchnorm/ReadVariableOp_1:value:0(batch_normalization_52/batchnorm/mul:z:0*
T0*
_output_shapes
:¨
1batch_normalization_52/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_52_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0·
$batch_normalization_52/batchnorm/subSub9batch_normalization_52/batchnorm/ReadVariableOp_2:value:0*batch_normalization_52/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_52/batchnorm/add_1AddV2*batch_normalization_52/batchnorm/mul_1:z:0(batch_normalization_52/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_52/LeakyRelu	LeakyRelu*batch_normalization_52/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_59/MatMulMatMul&leaky_re_lu_52/LeakyRelu:activations:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/batch_normalization_53/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_53_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_53/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¼
$batch_normalization_53/batchnorm/addAddV27batch_normalization_53/batchnorm/ReadVariableOp:value:0/batch_normalization_53/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_53/batchnorm/RsqrtRsqrt(batch_normalization_53/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_53/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_53_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_53/batchnorm/mulMul*batch_normalization_53/batchnorm/Rsqrt:y:0;batch_normalization_53/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¤
&batch_normalization_53/batchnorm/mul_1Muldense_59/BiasAdd:output:0(batch_normalization_53/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
1batch_normalization_53/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_53_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0·
&batch_normalization_53/batchnorm/mul_2Mul9batch_normalization_53/batchnorm/ReadVariableOp_1:value:0(batch_normalization_53/batchnorm/mul:z:0*
T0*
_output_shapes
:¨
1batch_normalization_53/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_53_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0·
$batch_normalization_53/batchnorm/subSub9batch_normalization_53/batchnorm/ReadVariableOp_2:value:0*batch_normalization_53/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_53/batchnorm/add_1AddV2*batch_normalization_53/batchnorm/mul_1:z:0(batch_normalization_53/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_53/LeakyRelu	LeakyRelu*batch_normalization_53/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_60/MatMulMatMul&leaky_re_lu_53/LeakyRelu:activations:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/batch_normalization_54/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_54_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_54/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¼
$batch_normalization_54/batchnorm/addAddV27batch_normalization_54/batchnorm/ReadVariableOp:value:0/batch_normalization_54/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_54/batchnorm/RsqrtRsqrt(batch_normalization_54/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_54/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_54_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_54/batchnorm/mulMul*batch_normalization_54/batchnorm/Rsqrt:y:0;batch_normalization_54/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¤
&batch_normalization_54/batchnorm/mul_1Muldense_60/BiasAdd:output:0(batch_normalization_54/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
1batch_normalization_54/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_54_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0·
&batch_normalization_54/batchnorm/mul_2Mul9batch_normalization_54/batchnorm/ReadVariableOp_1:value:0(batch_normalization_54/batchnorm/mul:z:0*
T0*
_output_shapes
:¨
1batch_normalization_54/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_54_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0·
$batch_normalization_54/batchnorm/subSub9batch_normalization_54/batchnorm/ReadVariableOp_2:value:0*batch_normalization_54/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_54/batchnorm/add_1AddV2*batch_normalization_54/batchnorm/mul_1:z:0(batch_normalization_54/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_54/LeakyRelu	LeakyRelu*batch_normalization_54/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_61/MatMulMatMul&leaky_re_lu_54/LeakyRelu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/batch_normalization_55/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_55_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_55/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¼
$batch_normalization_55/batchnorm/addAddV27batch_normalization_55/batchnorm/ReadVariableOp:value:0/batch_normalization_55/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_55/batchnorm/RsqrtRsqrt(batch_normalization_55/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_55/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_55_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_55/batchnorm/mulMul*batch_normalization_55/batchnorm/Rsqrt:y:0;batch_normalization_55/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¤
&batch_normalization_55/batchnorm/mul_1Muldense_61/BiasAdd:output:0(batch_normalization_55/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
1batch_normalization_55/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_55_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0·
&batch_normalization_55/batchnorm/mul_2Mul9batch_normalization_55/batchnorm/ReadVariableOp_1:value:0(batch_normalization_55/batchnorm/mul:z:0*
T0*
_output_shapes
:¨
1batch_normalization_55/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_55_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0·
$batch_normalization_55/batchnorm/subSub9batch_normalization_55/batchnorm/ReadVariableOp_2:value:0*batch_normalization_55/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_55/batchnorm/add_1AddV2*batch_normalization_55/batchnorm/mul_1:z:0(batch_normalization_55/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_55/LeakyRelu	LeakyRelu*batch_normalization_55/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_62/MatMulMatMul&leaky_re_lu_55/LeakyRelu:activations:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_62/BiasAdd/ReadVariableOpReadVariableOp(dense_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_62/BiasAddBiasAdddense_62/MatMul:product:0'dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.dense_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

:7*
dtype0
dense_55/kernel/Regularizer/AbsAbs6dense_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7r
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_55/kernel/Regularizer/SumSum#dense_55/kernel/Regularizer/Abs:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_56/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource*
_output_shapes

:77*
dtype0
dense_56/kernel/Regularizer/AbsAbs6dense_56/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77r
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum#dense_56/kernel/Regularizer/Abs:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_57/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes

:7u*
dtype0
dense_57/kernel/Regularizer/AbsAbs6dense_57/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7ur
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum#dense_57/kernel/Regularizer/Abs:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *û9);
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_58/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

:u*
dtype0
dense_58/kernel/Regularizer/AbsAbs6dense_58/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:ur
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum#dense_58/kernel/Regularizer/Abs:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_59/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_59/kernel/Regularizer/AbsAbs6dense_59/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum#dense_59/kernel/Regularizer/Abs:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_60/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_60/kernel/Regularizer/AbsAbs6dense_60/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_60/kernel/Regularizer/SumSum#dense_60/kernel/Regularizer/Abs:y:0*dense_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_60/kernel/Regularizer/mulMul*dense_60/kernel/Regularizer/mul/x:output:0(dense_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_61/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_61/kernel/Regularizer/AbsAbs6dense_61/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_61/kernel/Regularizer/SumSum#dense_61/kernel/Regularizer/Abs:y:0*dense_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_61/kernel/Regularizer/mulMul*dense_61/kernel/Regularizer/mul/x:output:0(dense_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_62/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
NoOpNoOp0^batch_normalization_49/batchnorm/ReadVariableOp2^batch_normalization_49/batchnorm/ReadVariableOp_12^batch_normalization_49/batchnorm/ReadVariableOp_24^batch_normalization_49/batchnorm/mul/ReadVariableOp0^batch_normalization_50/batchnorm/ReadVariableOp2^batch_normalization_50/batchnorm/ReadVariableOp_12^batch_normalization_50/batchnorm/ReadVariableOp_24^batch_normalization_50/batchnorm/mul/ReadVariableOp0^batch_normalization_51/batchnorm/ReadVariableOp2^batch_normalization_51/batchnorm/ReadVariableOp_12^batch_normalization_51/batchnorm/ReadVariableOp_24^batch_normalization_51/batchnorm/mul/ReadVariableOp0^batch_normalization_52/batchnorm/ReadVariableOp2^batch_normalization_52/batchnorm/ReadVariableOp_12^batch_normalization_52/batchnorm/ReadVariableOp_24^batch_normalization_52/batchnorm/mul/ReadVariableOp0^batch_normalization_53/batchnorm/ReadVariableOp2^batch_normalization_53/batchnorm/ReadVariableOp_12^batch_normalization_53/batchnorm/ReadVariableOp_24^batch_normalization_53/batchnorm/mul/ReadVariableOp0^batch_normalization_54/batchnorm/ReadVariableOp2^batch_normalization_54/batchnorm/ReadVariableOp_12^batch_normalization_54/batchnorm/ReadVariableOp_24^batch_normalization_54/batchnorm/mul/ReadVariableOp0^batch_normalization_55/batchnorm/ReadVariableOp2^batch_normalization_55/batchnorm/ReadVariableOp_12^batch_normalization_55/batchnorm/ReadVariableOp_24^batch_normalization_55/batchnorm/mul/ReadVariableOp ^dense_55/BiasAdd/ReadVariableOp^dense_55/MatMul/ReadVariableOp/^dense_55/kernel/Regularizer/Abs/ReadVariableOp ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp/^dense_56/kernel/Regularizer/Abs/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp/^dense_57/kernel/Regularizer/Abs/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp/^dense_58/kernel/Regularizer/Abs/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp/^dense_59/kernel/Regularizer/Abs/ReadVariableOp ^dense_60/BiasAdd/ReadVariableOp^dense_60/MatMul/ReadVariableOp/^dense_60/kernel/Regularizer/Abs/ReadVariableOp ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp/^dense_61/kernel/Regularizer/Abs/ReadVariableOp ^dense_62/BiasAdd/ReadVariableOp^dense_62/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_49/batchnorm/ReadVariableOp/batch_normalization_49/batchnorm/ReadVariableOp2f
1batch_normalization_49/batchnorm/ReadVariableOp_11batch_normalization_49/batchnorm/ReadVariableOp_12f
1batch_normalization_49/batchnorm/ReadVariableOp_21batch_normalization_49/batchnorm/ReadVariableOp_22j
3batch_normalization_49/batchnorm/mul/ReadVariableOp3batch_normalization_49/batchnorm/mul/ReadVariableOp2b
/batch_normalization_50/batchnorm/ReadVariableOp/batch_normalization_50/batchnorm/ReadVariableOp2f
1batch_normalization_50/batchnorm/ReadVariableOp_11batch_normalization_50/batchnorm/ReadVariableOp_12f
1batch_normalization_50/batchnorm/ReadVariableOp_21batch_normalization_50/batchnorm/ReadVariableOp_22j
3batch_normalization_50/batchnorm/mul/ReadVariableOp3batch_normalization_50/batchnorm/mul/ReadVariableOp2b
/batch_normalization_51/batchnorm/ReadVariableOp/batch_normalization_51/batchnorm/ReadVariableOp2f
1batch_normalization_51/batchnorm/ReadVariableOp_11batch_normalization_51/batchnorm/ReadVariableOp_12f
1batch_normalization_51/batchnorm/ReadVariableOp_21batch_normalization_51/batchnorm/ReadVariableOp_22j
3batch_normalization_51/batchnorm/mul/ReadVariableOp3batch_normalization_51/batchnorm/mul/ReadVariableOp2b
/batch_normalization_52/batchnorm/ReadVariableOp/batch_normalization_52/batchnorm/ReadVariableOp2f
1batch_normalization_52/batchnorm/ReadVariableOp_11batch_normalization_52/batchnorm/ReadVariableOp_12f
1batch_normalization_52/batchnorm/ReadVariableOp_21batch_normalization_52/batchnorm/ReadVariableOp_22j
3batch_normalization_52/batchnorm/mul/ReadVariableOp3batch_normalization_52/batchnorm/mul/ReadVariableOp2b
/batch_normalization_53/batchnorm/ReadVariableOp/batch_normalization_53/batchnorm/ReadVariableOp2f
1batch_normalization_53/batchnorm/ReadVariableOp_11batch_normalization_53/batchnorm/ReadVariableOp_12f
1batch_normalization_53/batchnorm/ReadVariableOp_21batch_normalization_53/batchnorm/ReadVariableOp_22j
3batch_normalization_53/batchnorm/mul/ReadVariableOp3batch_normalization_53/batchnorm/mul/ReadVariableOp2b
/batch_normalization_54/batchnorm/ReadVariableOp/batch_normalization_54/batchnorm/ReadVariableOp2f
1batch_normalization_54/batchnorm/ReadVariableOp_11batch_normalization_54/batchnorm/ReadVariableOp_12f
1batch_normalization_54/batchnorm/ReadVariableOp_21batch_normalization_54/batchnorm/ReadVariableOp_22j
3batch_normalization_54/batchnorm/mul/ReadVariableOp3batch_normalization_54/batchnorm/mul/ReadVariableOp2b
/batch_normalization_55/batchnorm/ReadVariableOp/batch_normalization_55/batchnorm/ReadVariableOp2f
1batch_normalization_55/batchnorm/ReadVariableOp_11batch_normalization_55/batchnorm/ReadVariableOp_12f
1batch_normalization_55/batchnorm/ReadVariableOp_21batch_normalization_55/batchnorm/ReadVariableOp_22j
3batch_normalization_55/batchnorm/mul/ReadVariableOp3batch_normalization_55/batchnorm/mul/ReadVariableOp2B
dense_55/BiasAdd/ReadVariableOpdense_55/BiasAdd/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp2`
.dense_55/kernel/Regularizer/Abs/ReadVariableOp.dense_55/kernel/Regularizer/Abs/ReadVariableOp2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2`
.dense_56/kernel/Regularizer/Abs/ReadVariableOp.dense_56/kernel/Regularizer/Abs/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2`
.dense_57/kernel/Regularizer/Abs/ReadVariableOp.dense_57/kernel/Regularizer/Abs/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2`
.dense_58/kernel/Regularizer/Abs/ReadVariableOp.dense_58/kernel/Regularizer/Abs/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp2`
.dense_59/kernel/Regularizer/Abs/ReadVariableOp.dense_59/kernel/Regularizer/Abs/ReadVariableOp2B
dense_60/BiasAdd/ReadVariableOpdense_60/BiasAdd/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2`
.dense_60/kernel/Regularizer/Abs/ReadVariableOp.dense_60/kernel/Regularizer/Abs/ReadVariableOp2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2`
.dense_61/kernel/Regularizer/Abs/ReadVariableOp.dense_61/kernel/Regularizer/Abs/ReadVariableOp2B
dense_62/BiasAdd/ReadVariableOpdense_62/BiasAdd/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ð
²
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1766820

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
¾
§
E__inference_dense_56_layer_call_and_return_conditional_losses_1769782

inputs0
matmul_readvariableop_resource:77-
biasadd_readvariableop_resource:7
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense_56/kernel/Regularizer/Abs/ReadVariableOpt
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
:ÿÿÿÿÿÿÿÿÿ7
.dense_56/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:77*
dtype0
dense_56/kernel/Regularizer/AbsAbs6dense_56/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77r
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum#dense_56/kernel/Regularizer/Abs:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¨
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_56/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_56/kernel/Regularizer/Abs/ReadVariableOp.dense_56/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
¾
§
E__inference_dense_61_layer_call_and_return_conditional_losses_1770387

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense_61/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.dense_61/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_61/kernel/Regularizer/AbsAbs6dense_61/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_61/kernel/Regularizer/SumSum#dense_61/kernel/Regularizer/Abs:y:0*dense_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_61/kernel/Regularizer/mulMul*dense_61/kernel/Regularizer/mul/x:output:0(dense_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_61/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_61/kernel/Regularizer/Abs/ReadVariableOp.dense_61/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_62_layer_call_and_return_conditional_losses_1770496

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

*__inference_dense_56_layer_call_fn_1769766

inputs
unknown:77
	unknown_0:7
identity¢StatefulPartitionedCallÚ
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
GPU 2J 8 *N
fIRG
E__inference_dense_56_layer_call_and_return_conditional_losses_1767438o
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
«
L
0__inference_leaky_re_lu_54_layer_call_fn_1770351

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_1767610`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_51_layer_call_fn_1769916

inputs
unknown:u
	unknown_0:u
	unknown_1:u
	unknown_2:u
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1766984o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿu: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_1767572

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
íç
Æ0
I__inference_sequential_6_layer_call_and_return_conditional_losses_1769484

inputs
normalization_6_sub_y
normalization_6_sqrt_x9
'dense_55_matmul_readvariableop_resource:76
(dense_55_biasadd_readvariableop_resource:7L
>batch_normalization_49_assignmovingavg_readvariableop_resource:7N
@batch_normalization_49_assignmovingavg_1_readvariableop_resource:7J
<batch_normalization_49_batchnorm_mul_readvariableop_resource:7F
8batch_normalization_49_batchnorm_readvariableop_resource:79
'dense_56_matmul_readvariableop_resource:776
(dense_56_biasadd_readvariableop_resource:7L
>batch_normalization_50_assignmovingavg_readvariableop_resource:7N
@batch_normalization_50_assignmovingavg_1_readvariableop_resource:7J
<batch_normalization_50_batchnorm_mul_readvariableop_resource:7F
8batch_normalization_50_batchnorm_readvariableop_resource:79
'dense_57_matmul_readvariableop_resource:7u6
(dense_57_biasadd_readvariableop_resource:uL
>batch_normalization_51_assignmovingavg_readvariableop_resource:uN
@batch_normalization_51_assignmovingavg_1_readvariableop_resource:uJ
<batch_normalization_51_batchnorm_mul_readvariableop_resource:uF
8batch_normalization_51_batchnorm_readvariableop_resource:u9
'dense_58_matmul_readvariableop_resource:u6
(dense_58_biasadd_readvariableop_resource:L
>batch_normalization_52_assignmovingavg_readvariableop_resource:N
@batch_normalization_52_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_52_batchnorm_mul_readvariableop_resource:F
8batch_normalization_52_batchnorm_readvariableop_resource:9
'dense_59_matmul_readvariableop_resource:6
(dense_59_biasadd_readvariableop_resource:L
>batch_normalization_53_assignmovingavg_readvariableop_resource:N
@batch_normalization_53_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_53_batchnorm_mul_readvariableop_resource:F
8batch_normalization_53_batchnorm_readvariableop_resource:9
'dense_60_matmul_readvariableop_resource:6
(dense_60_biasadd_readvariableop_resource:L
>batch_normalization_54_assignmovingavg_readvariableop_resource:N
@batch_normalization_54_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_54_batchnorm_mul_readvariableop_resource:F
8batch_normalization_54_batchnorm_readvariableop_resource:9
'dense_61_matmul_readvariableop_resource:6
(dense_61_biasadd_readvariableop_resource:L
>batch_normalization_55_assignmovingavg_readvariableop_resource:N
@batch_normalization_55_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_55_batchnorm_mul_readvariableop_resource:F
8batch_normalization_55_batchnorm_readvariableop_resource:9
'dense_62_matmul_readvariableop_resource:6
(dense_62_biasadd_readvariableop_resource:
identity¢&batch_normalization_49/AssignMovingAvg¢5batch_normalization_49/AssignMovingAvg/ReadVariableOp¢(batch_normalization_49/AssignMovingAvg_1¢7batch_normalization_49/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_49/batchnorm/ReadVariableOp¢3batch_normalization_49/batchnorm/mul/ReadVariableOp¢&batch_normalization_50/AssignMovingAvg¢5batch_normalization_50/AssignMovingAvg/ReadVariableOp¢(batch_normalization_50/AssignMovingAvg_1¢7batch_normalization_50/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_50/batchnorm/ReadVariableOp¢3batch_normalization_50/batchnorm/mul/ReadVariableOp¢&batch_normalization_51/AssignMovingAvg¢5batch_normalization_51/AssignMovingAvg/ReadVariableOp¢(batch_normalization_51/AssignMovingAvg_1¢7batch_normalization_51/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_51/batchnorm/ReadVariableOp¢3batch_normalization_51/batchnorm/mul/ReadVariableOp¢&batch_normalization_52/AssignMovingAvg¢5batch_normalization_52/AssignMovingAvg/ReadVariableOp¢(batch_normalization_52/AssignMovingAvg_1¢7batch_normalization_52/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_52/batchnorm/ReadVariableOp¢3batch_normalization_52/batchnorm/mul/ReadVariableOp¢&batch_normalization_53/AssignMovingAvg¢5batch_normalization_53/AssignMovingAvg/ReadVariableOp¢(batch_normalization_53/AssignMovingAvg_1¢7batch_normalization_53/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_53/batchnorm/ReadVariableOp¢3batch_normalization_53/batchnorm/mul/ReadVariableOp¢&batch_normalization_54/AssignMovingAvg¢5batch_normalization_54/AssignMovingAvg/ReadVariableOp¢(batch_normalization_54/AssignMovingAvg_1¢7batch_normalization_54/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_54/batchnorm/ReadVariableOp¢3batch_normalization_54/batchnorm/mul/ReadVariableOp¢&batch_normalization_55/AssignMovingAvg¢5batch_normalization_55/AssignMovingAvg/ReadVariableOp¢(batch_normalization_55/AssignMovingAvg_1¢7batch_normalization_55/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_55/batchnorm/ReadVariableOp¢3batch_normalization_55/batchnorm/mul/ReadVariableOp¢dense_55/BiasAdd/ReadVariableOp¢dense_55/MatMul/ReadVariableOp¢.dense_55/kernel/Regularizer/Abs/ReadVariableOp¢dense_56/BiasAdd/ReadVariableOp¢dense_56/MatMul/ReadVariableOp¢.dense_56/kernel/Regularizer/Abs/ReadVariableOp¢dense_57/BiasAdd/ReadVariableOp¢dense_57/MatMul/ReadVariableOp¢.dense_57/kernel/Regularizer/Abs/ReadVariableOp¢dense_58/BiasAdd/ReadVariableOp¢dense_58/MatMul/ReadVariableOp¢.dense_58/kernel/Regularizer/Abs/ReadVariableOp¢dense_59/BiasAdd/ReadVariableOp¢dense_59/MatMul/ReadVariableOp¢.dense_59/kernel/Regularizer/Abs/ReadVariableOp¢dense_60/BiasAdd/ReadVariableOp¢dense_60/MatMul/ReadVariableOp¢.dense_60/kernel/Regularizer/Abs/ReadVariableOp¢dense_61/BiasAdd/ReadVariableOp¢dense_61/MatMul/ReadVariableOp¢.dense_61/kernel/Regularizer/Abs/ReadVariableOp¢dense_62/BiasAdd/ReadVariableOp¢dense_62/MatMul/ReadVariableOpk
normalization_6/subSubinputsnormalization_6_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
normalization_6/SqrtSqrtnormalization_6_sqrt_x*
T0*
_output_shapes

:^
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

:7*
dtype0
dense_55/MatMulMatMulnormalization_6/truediv:z:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
dense_55/BiasAdd/ReadVariableOpReadVariableOp(dense_55_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
5batch_normalization_49/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: À
#batch_normalization_49/moments/meanMeandense_55/BiasAdd:output:0>batch_normalization_49/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(
+batch_normalization_49/moments/StopGradientStopGradient,batch_normalization_49/moments/mean:output:0*
T0*
_output_shapes

:7È
0batch_normalization_49/moments/SquaredDifferenceSquaredDifferencedense_55/BiasAdd:output:04batch_normalization_49/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
9batch_normalization_49/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ã
'batch_normalization_49/moments/varianceMean4batch_normalization_49/moments/SquaredDifference:z:0Bbatch_normalization_49/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(
&batch_normalization_49/moments/SqueezeSqueeze,batch_normalization_49/moments/mean:output:0*
T0*
_output_shapes
:7*
squeeze_dims
 ¡
(batch_normalization_49/moments/Squeeze_1Squeeze0batch_normalization_49/moments/variance:output:0*
T0*
_output_shapes
:7*
squeeze_dims
 q
,batch_normalization_49/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<°
5batch_normalization_49/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_49_assignmovingavg_readvariableop_resource*
_output_shapes
:7*
dtype0Æ
*batch_normalization_49/AssignMovingAvg/subSub=batch_normalization_49/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_49/moments/Squeeze:output:0*
T0*
_output_shapes
:7½
*batch_normalization_49/AssignMovingAvg/mulMul.batch_normalization_49/AssignMovingAvg/sub:z:05batch_normalization_49/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:7
&batch_normalization_49/AssignMovingAvgAssignSubVariableOp>batch_normalization_49_assignmovingavg_readvariableop_resource.batch_normalization_49/AssignMovingAvg/mul:z:06^batch_normalization_49/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_49/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<´
7batch_normalization_49/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_49_assignmovingavg_1_readvariableop_resource*
_output_shapes
:7*
dtype0Ì
,batch_normalization_49/AssignMovingAvg_1/subSub?batch_normalization_49/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_49/moments/Squeeze_1:output:0*
T0*
_output_shapes
:7Ã
,batch_normalization_49/AssignMovingAvg_1/mulMul0batch_normalization_49/AssignMovingAvg_1/sub:z:07batch_normalization_49/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:7
(batch_normalization_49/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_49_assignmovingavg_1_readvariableop_resource0batch_normalization_49/AssignMovingAvg_1/mul:z:08^batch_normalization_49/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_49/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
$batch_normalization_49/batchnorm/addAddV21batch_normalization_49/moments/Squeeze_1:output:0/batch_normalization_49/batchnorm/add/y:output:0*
T0*
_output_shapes
:7~
&batch_normalization_49/batchnorm/RsqrtRsqrt(batch_normalization_49/batchnorm/add:z:0*
T0*
_output_shapes
:7¬
3batch_normalization_49/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_49_batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0¹
$batch_normalization_49/batchnorm/mulMul*batch_normalization_49/batchnorm/Rsqrt:y:0;batch_normalization_49/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7¤
&batch_normalization_49/batchnorm/mul_1Muldense_55/BiasAdd:output:0(batch_normalization_49/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7­
&batch_normalization_49/batchnorm/mul_2Mul/batch_normalization_49/moments/Squeeze:output:0(batch_normalization_49/batchnorm/mul:z:0*
T0*
_output_shapes
:7¤
/batch_normalization_49/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_49_batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0µ
$batch_normalization_49/batchnorm/subSub7batch_normalization_49/batchnorm/ReadVariableOp:value:0*batch_normalization_49/batchnorm/mul_2:z:0*
T0*
_output_shapes
:7·
&batch_normalization_49/batchnorm/add_1AddV2*batch_normalization_49/batchnorm/mul_1:z:0(batch_normalization_49/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
leaky_re_lu_49/LeakyRelu	LeakyRelu*batch_normalization_49/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource*
_output_shapes

:77*
dtype0
dense_56/MatMulMatMul&leaky_re_lu_49/LeakyRelu:activations:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
dense_56/BiasAdd/ReadVariableOpReadVariableOp(dense_56_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
5batch_normalization_50/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: À
#batch_normalization_50/moments/meanMeandense_56/BiasAdd:output:0>batch_normalization_50/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(
+batch_normalization_50/moments/StopGradientStopGradient,batch_normalization_50/moments/mean:output:0*
T0*
_output_shapes

:7È
0batch_normalization_50/moments/SquaredDifferenceSquaredDifferencedense_56/BiasAdd:output:04batch_normalization_50/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
9batch_normalization_50/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ã
'batch_normalization_50/moments/varianceMean4batch_normalization_50/moments/SquaredDifference:z:0Bbatch_normalization_50/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(
&batch_normalization_50/moments/SqueezeSqueeze,batch_normalization_50/moments/mean:output:0*
T0*
_output_shapes
:7*
squeeze_dims
 ¡
(batch_normalization_50/moments/Squeeze_1Squeeze0batch_normalization_50/moments/variance:output:0*
T0*
_output_shapes
:7*
squeeze_dims
 q
,batch_normalization_50/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<°
5batch_normalization_50/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_50_assignmovingavg_readvariableop_resource*
_output_shapes
:7*
dtype0Æ
*batch_normalization_50/AssignMovingAvg/subSub=batch_normalization_50/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_50/moments/Squeeze:output:0*
T0*
_output_shapes
:7½
*batch_normalization_50/AssignMovingAvg/mulMul.batch_normalization_50/AssignMovingAvg/sub:z:05batch_normalization_50/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:7
&batch_normalization_50/AssignMovingAvgAssignSubVariableOp>batch_normalization_50_assignmovingavg_readvariableop_resource.batch_normalization_50/AssignMovingAvg/mul:z:06^batch_normalization_50/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_50/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<´
7batch_normalization_50/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_50_assignmovingavg_1_readvariableop_resource*
_output_shapes
:7*
dtype0Ì
,batch_normalization_50/AssignMovingAvg_1/subSub?batch_normalization_50/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_50/moments/Squeeze_1:output:0*
T0*
_output_shapes
:7Ã
,batch_normalization_50/AssignMovingAvg_1/mulMul0batch_normalization_50/AssignMovingAvg_1/sub:z:07batch_normalization_50/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:7
(batch_normalization_50/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_50_assignmovingavg_1_readvariableop_resource0batch_normalization_50/AssignMovingAvg_1/mul:z:08^batch_normalization_50/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_50/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
$batch_normalization_50/batchnorm/addAddV21batch_normalization_50/moments/Squeeze_1:output:0/batch_normalization_50/batchnorm/add/y:output:0*
T0*
_output_shapes
:7~
&batch_normalization_50/batchnorm/RsqrtRsqrt(batch_normalization_50/batchnorm/add:z:0*
T0*
_output_shapes
:7¬
3batch_normalization_50/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_50_batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0¹
$batch_normalization_50/batchnorm/mulMul*batch_normalization_50/batchnorm/Rsqrt:y:0;batch_normalization_50/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7¤
&batch_normalization_50/batchnorm/mul_1Muldense_56/BiasAdd:output:0(batch_normalization_50/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7­
&batch_normalization_50/batchnorm/mul_2Mul/batch_normalization_50/moments/Squeeze:output:0(batch_normalization_50/batchnorm/mul:z:0*
T0*
_output_shapes
:7¤
/batch_normalization_50/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_50_batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0µ
$batch_normalization_50/batchnorm/subSub7batch_normalization_50/batchnorm/ReadVariableOp:value:0*batch_normalization_50/batchnorm/mul_2:z:0*
T0*
_output_shapes
:7·
&batch_normalization_50/batchnorm/add_1AddV2*batch_normalization_50/batchnorm/mul_1:z:0(batch_normalization_50/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
leaky_re_lu_50/LeakyRelu	LeakyRelu*batch_normalization_50/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes

:7u*
dtype0
dense_57/MatMulMatMul&leaky_re_lu_50/LeakyRelu:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:u*
dtype0
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
5batch_normalization_51/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: À
#batch_normalization_51/moments/meanMeandense_57/BiasAdd:output:0>batch_normalization_51/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:u*
	keep_dims(
+batch_normalization_51/moments/StopGradientStopGradient,batch_normalization_51/moments/mean:output:0*
T0*
_output_shapes

:uÈ
0batch_normalization_51/moments/SquaredDifferenceSquaredDifferencedense_57/BiasAdd:output:04batch_normalization_51/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
9batch_normalization_51/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ã
'batch_normalization_51/moments/varianceMean4batch_normalization_51/moments/SquaredDifference:z:0Bbatch_normalization_51/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:u*
	keep_dims(
&batch_normalization_51/moments/SqueezeSqueeze,batch_normalization_51/moments/mean:output:0*
T0*
_output_shapes
:u*
squeeze_dims
 ¡
(batch_normalization_51/moments/Squeeze_1Squeeze0batch_normalization_51/moments/variance:output:0*
T0*
_output_shapes
:u*
squeeze_dims
 q
,batch_normalization_51/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<°
5batch_normalization_51/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_51_assignmovingavg_readvariableop_resource*
_output_shapes
:u*
dtype0Æ
*batch_normalization_51/AssignMovingAvg/subSub=batch_normalization_51/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_51/moments/Squeeze:output:0*
T0*
_output_shapes
:u½
*batch_normalization_51/AssignMovingAvg/mulMul.batch_normalization_51/AssignMovingAvg/sub:z:05batch_normalization_51/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:u
&batch_normalization_51/AssignMovingAvgAssignSubVariableOp>batch_normalization_51_assignmovingavg_readvariableop_resource.batch_normalization_51/AssignMovingAvg/mul:z:06^batch_normalization_51/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_51/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<´
7batch_normalization_51/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_51_assignmovingavg_1_readvariableop_resource*
_output_shapes
:u*
dtype0Ì
,batch_normalization_51/AssignMovingAvg_1/subSub?batch_normalization_51/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_51/moments/Squeeze_1:output:0*
T0*
_output_shapes
:uÃ
,batch_normalization_51/AssignMovingAvg_1/mulMul0batch_normalization_51/AssignMovingAvg_1/sub:z:07batch_normalization_51/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:u
(batch_normalization_51/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_51_assignmovingavg_1_readvariableop_resource0batch_normalization_51/AssignMovingAvg_1/mul:z:08^batch_normalization_51/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_51/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
$batch_normalization_51/batchnorm/addAddV21batch_normalization_51/moments/Squeeze_1:output:0/batch_normalization_51/batchnorm/add/y:output:0*
T0*
_output_shapes
:u~
&batch_normalization_51/batchnorm/RsqrtRsqrt(batch_normalization_51/batchnorm/add:z:0*
T0*
_output_shapes
:u¬
3batch_normalization_51/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_51_batchnorm_mul_readvariableop_resource*
_output_shapes
:u*
dtype0¹
$batch_normalization_51/batchnorm/mulMul*batch_normalization_51/batchnorm/Rsqrt:y:0;batch_normalization_51/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:u¤
&batch_normalization_51/batchnorm/mul_1Muldense_57/BiasAdd:output:0(batch_normalization_51/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu­
&batch_normalization_51/batchnorm/mul_2Mul/batch_normalization_51/moments/Squeeze:output:0(batch_normalization_51/batchnorm/mul:z:0*
T0*
_output_shapes
:u¤
/batch_normalization_51/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_51_batchnorm_readvariableop_resource*
_output_shapes
:u*
dtype0µ
$batch_normalization_51/batchnorm/subSub7batch_normalization_51/batchnorm/ReadVariableOp:value:0*batch_normalization_51/batchnorm/mul_2:z:0*
T0*
_output_shapes
:u·
&batch_normalization_51/batchnorm/add_1AddV2*batch_normalization_51/batchnorm/mul_1:z:0(batch_normalization_51/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
leaky_re_lu_51/LeakyRelu	LeakyRelu*batch_normalization_51/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*
alpha%>
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

:u*
dtype0
dense_58/MatMulMatMul&leaky_re_lu_51/LeakyRelu:activations:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_58/BiasAdd/ReadVariableOpReadVariableOp(dense_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5batch_normalization_52/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: À
#batch_normalization_52/moments/meanMeandense_58/BiasAdd:output:0>batch_normalization_52/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
+batch_normalization_52/moments/StopGradientStopGradient,batch_normalization_52/moments/mean:output:0*
T0*
_output_shapes

:È
0batch_normalization_52/moments/SquaredDifferenceSquaredDifferencedense_58/BiasAdd:output:04batch_normalization_52/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
9batch_normalization_52/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ã
'batch_normalization_52/moments/varianceMean4batch_normalization_52/moments/SquaredDifference:z:0Bbatch_normalization_52/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
&batch_normalization_52/moments/SqueezeSqueeze,batch_normalization_52/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ¡
(batch_normalization_52/moments/Squeeze_1Squeeze0batch_normalization_52/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_52/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<°
5batch_normalization_52/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_52_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Æ
*batch_normalization_52/AssignMovingAvg/subSub=batch_normalization_52/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_52/moments/Squeeze:output:0*
T0*
_output_shapes
:½
*batch_normalization_52/AssignMovingAvg/mulMul.batch_normalization_52/AssignMovingAvg/sub:z:05batch_normalization_52/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
&batch_normalization_52/AssignMovingAvgAssignSubVariableOp>batch_normalization_52_assignmovingavg_readvariableop_resource.batch_normalization_52/AssignMovingAvg/mul:z:06^batch_normalization_52/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_52/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<´
7batch_normalization_52/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_52_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ì
,batch_normalization_52/AssignMovingAvg_1/subSub?batch_normalization_52/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_52/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Ã
,batch_normalization_52/AssignMovingAvg_1/mulMul0batch_normalization_52/AssignMovingAvg_1/sub:z:07batch_normalization_52/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
(batch_normalization_52/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_52_assignmovingavg_1_readvariableop_resource0batch_normalization_52/AssignMovingAvg_1/mul:z:08^batch_normalization_52/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_52/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
$batch_normalization_52/batchnorm/addAddV21batch_normalization_52/moments/Squeeze_1:output:0/batch_normalization_52/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_52/batchnorm/RsqrtRsqrt(batch_normalization_52/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_52/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_52_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_52/batchnorm/mulMul*batch_normalization_52/batchnorm/Rsqrt:y:0;batch_normalization_52/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¤
&batch_normalization_52/batchnorm/mul_1Muldense_58/BiasAdd:output:0(batch_normalization_52/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
&batch_normalization_52/batchnorm/mul_2Mul/batch_normalization_52/moments/Squeeze:output:0(batch_normalization_52/batchnorm/mul:z:0*
T0*
_output_shapes
:¤
/batch_normalization_52/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_52_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0µ
$batch_normalization_52/batchnorm/subSub7batch_normalization_52/batchnorm/ReadVariableOp:value:0*batch_normalization_52/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_52/batchnorm/add_1AddV2*batch_normalization_52/batchnorm/mul_1:z:0(batch_normalization_52/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_52/LeakyRelu	LeakyRelu*batch_normalization_52/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_59/MatMulMatMul&leaky_re_lu_52/LeakyRelu:activations:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5batch_normalization_53/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: À
#batch_normalization_53/moments/meanMeandense_59/BiasAdd:output:0>batch_normalization_53/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
+batch_normalization_53/moments/StopGradientStopGradient,batch_normalization_53/moments/mean:output:0*
T0*
_output_shapes

:È
0batch_normalization_53/moments/SquaredDifferenceSquaredDifferencedense_59/BiasAdd:output:04batch_normalization_53/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
9batch_normalization_53/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ã
'batch_normalization_53/moments/varianceMean4batch_normalization_53/moments/SquaredDifference:z:0Bbatch_normalization_53/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
&batch_normalization_53/moments/SqueezeSqueeze,batch_normalization_53/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ¡
(batch_normalization_53/moments/Squeeze_1Squeeze0batch_normalization_53/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_53/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<°
5batch_normalization_53/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_53_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Æ
*batch_normalization_53/AssignMovingAvg/subSub=batch_normalization_53/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_53/moments/Squeeze:output:0*
T0*
_output_shapes
:½
*batch_normalization_53/AssignMovingAvg/mulMul.batch_normalization_53/AssignMovingAvg/sub:z:05batch_normalization_53/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
&batch_normalization_53/AssignMovingAvgAssignSubVariableOp>batch_normalization_53_assignmovingavg_readvariableop_resource.batch_normalization_53/AssignMovingAvg/mul:z:06^batch_normalization_53/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_53/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<´
7batch_normalization_53/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_53_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ì
,batch_normalization_53/AssignMovingAvg_1/subSub?batch_normalization_53/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_53/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Ã
,batch_normalization_53/AssignMovingAvg_1/mulMul0batch_normalization_53/AssignMovingAvg_1/sub:z:07batch_normalization_53/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
(batch_normalization_53/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_53_assignmovingavg_1_readvariableop_resource0batch_normalization_53/AssignMovingAvg_1/mul:z:08^batch_normalization_53/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_53/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
$batch_normalization_53/batchnorm/addAddV21batch_normalization_53/moments/Squeeze_1:output:0/batch_normalization_53/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_53/batchnorm/RsqrtRsqrt(batch_normalization_53/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_53/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_53_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_53/batchnorm/mulMul*batch_normalization_53/batchnorm/Rsqrt:y:0;batch_normalization_53/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¤
&batch_normalization_53/batchnorm/mul_1Muldense_59/BiasAdd:output:0(batch_normalization_53/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
&batch_normalization_53/batchnorm/mul_2Mul/batch_normalization_53/moments/Squeeze:output:0(batch_normalization_53/batchnorm/mul:z:0*
T0*
_output_shapes
:¤
/batch_normalization_53/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_53_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0µ
$batch_normalization_53/batchnorm/subSub7batch_normalization_53/batchnorm/ReadVariableOp:value:0*batch_normalization_53/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_53/batchnorm/add_1AddV2*batch_normalization_53/batchnorm/mul_1:z:0(batch_normalization_53/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_53/LeakyRelu	LeakyRelu*batch_normalization_53/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_60/MatMul/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_60/MatMulMatMul&leaky_re_lu_53/LeakyRelu:activations:0&dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_60/BiasAdd/ReadVariableOpReadVariableOp(dense_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_60/BiasAddBiasAdddense_60/MatMul:product:0'dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5batch_normalization_54/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: À
#batch_normalization_54/moments/meanMeandense_60/BiasAdd:output:0>batch_normalization_54/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
+batch_normalization_54/moments/StopGradientStopGradient,batch_normalization_54/moments/mean:output:0*
T0*
_output_shapes

:È
0batch_normalization_54/moments/SquaredDifferenceSquaredDifferencedense_60/BiasAdd:output:04batch_normalization_54/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
9batch_normalization_54/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ã
'batch_normalization_54/moments/varianceMean4batch_normalization_54/moments/SquaredDifference:z:0Bbatch_normalization_54/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
&batch_normalization_54/moments/SqueezeSqueeze,batch_normalization_54/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ¡
(batch_normalization_54/moments/Squeeze_1Squeeze0batch_normalization_54/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_54/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<°
5batch_normalization_54/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_54_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Æ
*batch_normalization_54/AssignMovingAvg/subSub=batch_normalization_54/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_54/moments/Squeeze:output:0*
T0*
_output_shapes
:½
*batch_normalization_54/AssignMovingAvg/mulMul.batch_normalization_54/AssignMovingAvg/sub:z:05batch_normalization_54/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
&batch_normalization_54/AssignMovingAvgAssignSubVariableOp>batch_normalization_54_assignmovingavg_readvariableop_resource.batch_normalization_54/AssignMovingAvg/mul:z:06^batch_normalization_54/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_54/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<´
7batch_normalization_54/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_54_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ì
,batch_normalization_54/AssignMovingAvg_1/subSub?batch_normalization_54/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_54/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Ã
,batch_normalization_54/AssignMovingAvg_1/mulMul0batch_normalization_54/AssignMovingAvg_1/sub:z:07batch_normalization_54/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
(batch_normalization_54/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_54_assignmovingavg_1_readvariableop_resource0batch_normalization_54/AssignMovingAvg_1/mul:z:08^batch_normalization_54/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_54/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
$batch_normalization_54/batchnorm/addAddV21batch_normalization_54/moments/Squeeze_1:output:0/batch_normalization_54/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_54/batchnorm/RsqrtRsqrt(batch_normalization_54/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_54/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_54_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_54/batchnorm/mulMul*batch_normalization_54/batchnorm/Rsqrt:y:0;batch_normalization_54/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¤
&batch_normalization_54/batchnorm/mul_1Muldense_60/BiasAdd:output:0(batch_normalization_54/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
&batch_normalization_54/batchnorm/mul_2Mul/batch_normalization_54/moments/Squeeze:output:0(batch_normalization_54/batchnorm/mul:z:0*
T0*
_output_shapes
:¤
/batch_normalization_54/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_54_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0µ
$batch_normalization_54/batchnorm/subSub7batch_normalization_54/batchnorm/ReadVariableOp:value:0*batch_normalization_54/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_54/batchnorm/add_1AddV2*batch_normalization_54/batchnorm/mul_1:z:0(batch_normalization_54/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_54/LeakyRelu	LeakyRelu*batch_normalization_54/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_61/MatMulMatMul&leaky_re_lu_54/LeakyRelu:activations:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5batch_normalization_55/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: À
#batch_normalization_55/moments/meanMeandense_61/BiasAdd:output:0>batch_normalization_55/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
+batch_normalization_55/moments/StopGradientStopGradient,batch_normalization_55/moments/mean:output:0*
T0*
_output_shapes

:È
0batch_normalization_55/moments/SquaredDifferenceSquaredDifferencedense_61/BiasAdd:output:04batch_normalization_55/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
9batch_normalization_55/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ã
'batch_normalization_55/moments/varianceMean4batch_normalization_55/moments/SquaredDifference:z:0Bbatch_normalization_55/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
&batch_normalization_55/moments/SqueezeSqueeze,batch_normalization_55/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ¡
(batch_normalization_55/moments/Squeeze_1Squeeze0batch_normalization_55/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_55/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<°
5batch_normalization_55/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_55_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Æ
*batch_normalization_55/AssignMovingAvg/subSub=batch_normalization_55/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_55/moments/Squeeze:output:0*
T0*
_output_shapes
:½
*batch_normalization_55/AssignMovingAvg/mulMul.batch_normalization_55/AssignMovingAvg/sub:z:05batch_normalization_55/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
&batch_normalization_55/AssignMovingAvgAssignSubVariableOp>batch_normalization_55_assignmovingavg_readvariableop_resource.batch_normalization_55/AssignMovingAvg/mul:z:06^batch_normalization_55/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_55/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<´
7batch_normalization_55/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_55_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ì
,batch_normalization_55/AssignMovingAvg_1/subSub?batch_normalization_55/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_55/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Ã
,batch_normalization_55/AssignMovingAvg_1/mulMul0batch_normalization_55/AssignMovingAvg_1/sub:z:07batch_normalization_55/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
(batch_normalization_55/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_55_assignmovingavg_1_readvariableop_resource0batch_normalization_55/AssignMovingAvg_1/mul:z:08^batch_normalization_55/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_55/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
$batch_normalization_55/batchnorm/addAddV21batch_normalization_55/moments/Squeeze_1:output:0/batch_normalization_55/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_55/batchnorm/RsqrtRsqrt(batch_normalization_55/batchnorm/add:z:0*
T0*
_output_shapes
:¬
3batch_normalization_55/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_55_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¹
$batch_normalization_55/batchnorm/mulMul*batch_normalization_55/batchnorm/Rsqrt:y:0;batch_normalization_55/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¤
&batch_normalization_55/batchnorm/mul_1Muldense_61/BiasAdd:output:0(batch_normalization_55/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
&batch_normalization_55/batchnorm/mul_2Mul/batch_normalization_55/moments/Squeeze:output:0(batch_normalization_55/batchnorm/mul:z:0*
T0*
_output_shapes
:¤
/batch_normalization_55/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_55_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0µ
$batch_normalization_55/batchnorm/subSub7batch_normalization_55/batchnorm/ReadVariableOp:value:0*batch_normalization_55/batchnorm/mul_2:z:0*
T0*
_output_shapes
:·
&batch_normalization_55/batchnorm/add_1AddV2*batch_normalization_55/batchnorm/mul_1:z:0(batch_normalization_55/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_55/LeakyRelu	LeakyRelu*batch_normalization_55/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_62/MatMul/ReadVariableOpReadVariableOp'dense_62_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_62/MatMulMatMul&leaky_re_lu_55/LeakyRelu:activations:0&dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_62/BiasAdd/ReadVariableOpReadVariableOp(dense_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_62/BiasAddBiasAdddense_62/MatMul:product:0'dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.dense_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

:7*
dtype0
dense_55/kernel/Regularizer/AbsAbs6dense_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7r
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_55/kernel/Regularizer/SumSum#dense_55/kernel/Regularizer/Abs:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_56/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource*
_output_shapes

:77*
dtype0
dense_56/kernel/Regularizer/AbsAbs6dense_56/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77r
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum#dense_56/kernel/Regularizer/Abs:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_57/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes

:7u*
dtype0
dense_57/kernel/Regularizer/AbsAbs6dense_57/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7ur
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum#dense_57/kernel/Regularizer/Abs:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *û9);
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_58/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource*
_output_shapes

:u*
dtype0
dense_58/kernel/Regularizer/AbsAbs6dense_58/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:ur
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum#dense_58/kernel/Regularizer/Abs:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_59/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_59/kernel/Regularizer/AbsAbs6dense_59/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum#dense_59/kernel/Regularizer/Abs:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_60/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_60_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_60/kernel/Regularizer/AbsAbs6dense_60/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_60/kernel/Regularizer/SumSum#dense_60/kernel/Regularizer/Abs:y:0*dense_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_60/kernel/Regularizer/mulMul*dense_60/kernel/Regularizer/mul/x:output:0(dense_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_61/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_61/kernel/Regularizer/AbsAbs6dense_61/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_61/kernel/Regularizer/SumSum#dense_61/kernel/Regularizer/Abs:y:0*dense_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_61/kernel/Regularizer/mulMul*dense_61/kernel/Regularizer/mul/x:output:0(dense_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_62/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ÷
NoOpNoOp'^batch_normalization_49/AssignMovingAvg6^batch_normalization_49/AssignMovingAvg/ReadVariableOp)^batch_normalization_49/AssignMovingAvg_18^batch_normalization_49/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_49/batchnorm/ReadVariableOp4^batch_normalization_49/batchnorm/mul/ReadVariableOp'^batch_normalization_50/AssignMovingAvg6^batch_normalization_50/AssignMovingAvg/ReadVariableOp)^batch_normalization_50/AssignMovingAvg_18^batch_normalization_50/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_50/batchnorm/ReadVariableOp4^batch_normalization_50/batchnorm/mul/ReadVariableOp'^batch_normalization_51/AssignMovingAvg6^batch_normalization_51/AssignMovingAvg/ReadVariableOp)^batch_normalization_51/AssignMovingAvg_18^batch_normalization_51/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_51/batchnorm/ReadVariableOp4^batch_normalization_51/batchnorm/mul/ReadVariableOp'^batch_normalization_52/AssignMovingAvg6^batch_normalization_52/AssignMovingAvg/ReadVariableOp)^batch_normalization_52/AssignMovingAvg_18^batch_normalization_52/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_52/batchnorm/ReadVariableOp4^batch_normalization_52/batchnorm/mul/ReadVariableOp'^batch_normalization_53/AssignMovingAvg6^batch_normalization_53/AssignMovingAvg/ReadVariableOp)^batch_normalization_53/AssignMovingAvg_18^batch_normalization_53/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_53/batchnorm/ReadVariableOp4^batch_normalization_53/batchnorm/mul/ReadVariableOp'^batch_normalization_54/AssignMovingAvg6^batch_normalization_54/AssignMovingAvg/ReadVariableOp)^batch_normalization_54/AssignMovingAvg_18^batch_normalization_54/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_54/batchnorm/ReadVariableOp4^batch_normalization_54/batchnorm/mul/ReadVariableOp'^batch_normalization_55/AssignMovingAvg6^batch_normalization_55/AssignMovingAvg/ReadVariableOp)^batch_normalization_55/AssignMovingAvg_18^batch_normalization_55/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_55/batchnorm/ReadVariableOp4^batch_normalization_55/batchnorm/mul/ReadVariableOp ^dense_55/BiasAdd/ReadVariableOp^dense_55/MatMul/ReadVariableOp/^dense_55/kernel/Regularizer/Abs/ReadVariableOp ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp/^dense_56/kernel/Regularizer/Abs/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp/^dense_57/kernel/Regularizer/Abs/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp/^dense_58/kernel/Regularizer/Abs/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp/^dense_59/kernel/Regularizer/Abs/ReadVariableOp ^dense_60/BiasAdd/ReadVariableOp^dense_60/MatMul/ReadVariableOp/^dense_60/kernel/Regularizer/Abs/ReadVariableOp ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp/^dense_61/kernel/Regularizer/Abs/ReadVariableOp ^dense_62/BiasAdd/ReadVariableOp^dense_62/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_49/AssignMovingAvg&batch_normalization_49/AssignMovingAvg2n
5batch_normalization_49/AssignMovingAvg/ReadVariableOp5batch_normalization_49/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_49/AssignMovingAvg_1(batch_normalization_49/AssignMovingAvg_12r
7batch_normalization_49/AssignMovingAvg_1/ReadVariableOp7batch_normalization_49/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_49/batchnorm/ReadVariableOp/batch_normalization_49/batchnorm/ReadVariableOp2j
3batch_normalization_49/batchnorm/mul/ReadVariableOp3batch_normalization_49/batchnorm/mul/ReadVariableOp2P
&batch_normalization_50/AssignMovingAvg&batch_normalization_50/AssignMovingAvg2n
5batch_normalization_50/AssignMovingAvg/ReadVariableOp5batch_normalization_50/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_50/AssignMovingAvg_1(batch_normalization_50/AssignMovingAvg_12r
7batch_normalization_50/AssignMovingAvg_1/ReadVariableOp7batch_normalization_50/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_50/batchnorm/ReadVariableOp/batch_normalization_50/batchnorm/ReadVariableOp2j
3batch_normalization_50/batchnorm/mul/ReadVariableOp3batch_normalization_50/batchnorm/mul/ReadVariableOp2P
&batch_normalization_51/AssignMovingAvg&batch_normalization_51/AssignMovingAvg2n
5batch_normalization_51/AssignMovingAvg/ReadVariableOp5batch_normalization_51/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_51/AssignMovingAvg_1(batch_normalization_51/AssignMovingAvg_12r
7batch_normalization_51/AssignMovingAvg_1/ReadVariableOp7batch_normalization_51/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_51/batchnorm/ReadVariableOp/batch_normalization_51/batchnorm/ReadVariableOp2j
3batch_normalization_51/batchnorm/mul/ReadVariableOp3batch_normalization_51/batchnorm/mul/ReadVariableOp2P
&batch_normalization_52/AssignMovingAvg&batch_normalization_52/AssignMovingAvg2n
5batch_normalization_52/AssignMovingAvg/ReadVariableOp5batch_normalization_52/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_52/AssignMovingAvg_1(batch_normalization_52/AssignMovingAvg_12r
7batch_normalization_52/AssignMovingAvg_1/ReadVariableOp7batch_normalization_52/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_52/batchnorm/ReadVariableOp/batch_normalization_52/batchnorm/ReadVariableOp2j
3batch_normalization_52/batchnorm/mul/ReadVariableOp3batch_normalization_52/batchnorm/mul/ReadVariableOp2P
&batch_normalization_53/AssignMovingAvg&batch_normalization_53/AssignMovingAvg2n
5batch_normalization_53/AssignMovingAvg/ReadVariableOp5batch_normalization_53/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_53/AssignMovingAvg_1(batch_normalization_53/AssignMovingAvg_12r
7batch_normalization_53/AssignMovingAvg_1/ReadVariableOp7batch_normalization_53/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_53/batchnorm/ReadVariableOp/batch_normalization_53/batchnorm/ReadVariableOp2j
3batch_normalization_53/batchnorm/mul/ReadVariableOp3batch_normalization_53/batchnorm/mul/ReadVariableOp2P
&batch_normalization_54/AssignMovingAvg&batch_normalization_54/AssignMovingAvg2n
5batch_normalization_54/AssignMovingAvg/ReadVariableOp5batch_normalization_54/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_54/AssignMovingAvg_1(batch_normalization_54/AssignMovingAvg_12r
7batch_normalization_54/AssignMovingAvg_1/ReadVariableOp7batch_normalization_54/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_54/batchnorm/ReadVariableOp/batch_normalization_54/batchnorm/ReadVariableOp2j
3batch_normalization_54/batchnorm/mul/ReadVariableOp3batch_normalization_54/batchnorm/mul/ReadVariableOp2P
&batch_normalization_55/AssignMovingAvg&batch_normalization_55/AssignMovingAvg2n
5batch_normalization_55/AssignMovingAvg/ReadVariableOp5batch_normalization_55/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_55/AssignMovingAvg_1(batch_normalization_55/AssignMovingAvg_12r
7batch_normalization_55/AssignMovingAvg_1/ReadVariableOp7batch_normalization_55/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_55/batchnorm/ReadVariableOp/batch_normalization_55/batchnorm/ReadVariableOp2j
3batch_normalization_55/batchnorm/mul/ReadVariableOp3batch_normalization_55/batchnorm/mul/ReadVariableOp2B
dense_55/BiasAdd/ReadVariableOpdense_55/BiasAdd/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp2`
.dense_55/kernel/Regularizer/Abs/ReadVariableOp.dense_55/kernel/Regularizer/Abs/ReadVariableOp2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2`
.dense_56/kernel/Regularizer/Abs/ReadVariableOp.dense_56/kernel/Regularizer/Abs/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2`
.dense_57/kernel/Regularizer/Abs/ReadVariableOp.dense_57/kernel/Regularizer/Abs/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2`
.dense_58/kernel/Regularizer/Abs/ReadVariableOp.dense_58/kernel/Regularizer/Abs/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp2`
.dense_59/kernel/Regularizer/Abs/ReadVariableOp.dense_59/kernel/Regularizer/Abs/ReadVariableOp2B
dense_60/BiasAdd/ReadVariableOpdense_60/BiasAdd/ReadVariableOp2@
dense_60/MatMul/ReadVariableOpdense_60/MatMul/ReadVariableOp2`
.dense_60/kernel/Regularizer/Abs/ReadVariableOp.dense_60/kernel/Regularizer/Abs/ReadVariableOp2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2`
.dense_61/kernel/Regularizer/Abs/ReadVariableOp.dense_61/kernel/Regularizer/Abs/ReadVariableOp2B
dense_62/BiasAdd/ReadVariableOpdense_62/BiasAdd/ReadVariableOp2@
dense_62/MatMul/ReadVariableOpdense_62/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
¾
§
E__inference_dense_58_layer_call_and_return_conditional_losses_1770024

inputs0
matmul_readvariableop_resource:u-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense_58/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:u*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.dense_58/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:u*
dtype0
dense_58/kernel/Regularizer/AbsAbs6dense_58/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:ur
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum#dense_58/kernel/Regularizer/Abs:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_58/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿu: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_58/kernel/Regularizer/Abs/ReadVariableOp.dense_58/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_49_layer_call_fn_1769687

inputs
unknown:7
	unknown_0:7
	unknown_1:7
	unknown_2:7
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1766867o
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

¡

.__inference_sequential_6_layer_call_fn_1767804
normalization_6_input
unknown
	unknown_0
	unknown_1:7
	unknown_2:7
	unknown_3:7
	unknown_4:7
	unknown_5:7
	unknown_6:7
	unknown_7:77
	unknown_8:7
	unknown_9:7

unknown_10:7

unknown_11:7

unknown_12:7

unknown_13:7u

unknown_14:u

unknown_15:u

unknown_16:u

unknown_17:u

unknown_18:u

unknown_19:u

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallnormalization_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_1767709o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namenormalization_6_input:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_1767534

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
§
E__inference_dense_59_layer_call_and_return_conditional_losses_1767552

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense_59/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.dense_59/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_59/kernel/Regularizer/AbsAbs6dense_59/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum#dense_59/kernel/Regularizer/Abs:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_59/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_59/kernel/Regularizer/Abs/ReadVariableOp.dense_59/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¬
__inference_loss_fn_3_1770540I
7dense_58_kernel_regularizer_abs_readvariableop_resource:u
identity¢.dense_58/kernel/Regularizer/Abs/ReadVariableOp¦
.dense_58/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_58_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:u*
dtype0
dense_58/kernel/Regularizer/AbsAbs6dense_58/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:ur
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum#dense_58/kernel/Regularizer/Abs:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_58/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^dense_58/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_58/kernel/Regularizer/Abs/ReadVariableOp.dense_58/kernel/Regularizer/Abs/ReadVariableOp
%
ì
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1770104

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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
K__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_1767648

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
g
K__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_1770356

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_55_layer_call_fn_1770472

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_1767648`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¬
__inference_loss_fn_0_1770507I
7dense_55_kernel_regularizer_abs_readvariableop_resource:7
identity¢.dense_55/kernel/Regularizer/Abs/ReadVariableOp¦
.dense_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_55_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:7*
dtype0
dense_55/kernel/Regularizer/AbsAbs6dense_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7r
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_55/kernel/Regularizer/SumSum#dense_55/kernel/Regularizer/Abs:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_55/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^dense_55/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_55/kernel/Regularizer/Abs/ReadVariableOp.dense_55/kernel/Regularizer/Abs/ReadVariableOp
å
g
K__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_1770114

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
§
E__inference_dense_55_layer_call_and_return_conditional_losses_1767400

inputs0
matmul_readvariableop_resource:7-
biasadd_readvariableop_resource:7
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense_55/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7*
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
:ÿÿÿÿÿÿÿÿÿ7
.dense_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7*
dtype0
dense_55/kernel/Regularizer/AbsAbs6dense_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7r
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_55/kernel/Regularizer/SumSum#dense_55/kernel/Regularizer/Abs:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¨
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_55/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_55/kernel/Regularizer/Abs/ReadVariableOp.dense_55/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_51_layer_call_fn_1769988

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
:ÿÿÿÿÿÿÿÿÿu* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_1767496`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿu:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_53_layer_call_fn_1770158

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1767148o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

*__inference_dense_59_layer_call_fn_1770129

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_1767552o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
§
E__inference_dense_58_layer_call_and_return_conditional_losses_1767514

inputs0
matmul_readvariableop_resource:u-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense_58/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:u*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.dense_58/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:u*
dtype0
dense_58/kernel/Regularizer/AbsAbs6dense_58/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:ur
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum#dense_58/kernel/Regularizer/Abs:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_58/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿu: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_58/kernel/Regularizer/Abs/ReadVariableOp.dense_58/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_55_layer_call_fn_1770400

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1767312o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1766867

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
%
ì
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1770467

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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
«
L
0__inference_leaky_re_lu_49_layer_call_fn_1769746

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
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_1767420`
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
Ä

*__inference_dense_58_layer_call_fn_1770008

inputs
unknown:u
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_58_layer_call_and_return_conditional_losses_1767514o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿu: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 
_user_specified_nameinputs
¾
§
E__inference_dense_60_layer_call_and_return_conditional_losses_1770266

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense_60/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.dense_60/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_60/kernel/Regularizer/AbsAbs6dense_60/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_60/kernel/Regularizer/SumSum#dense_60/kernel/Regularizer/Abs:y:0*dense_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_60/kernel/Regularizer/mulMul*dense_60/kernel/Regularizer/mul/x:output:0(dense_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_60/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_60/kernel/Regularizer/Abs/ReadVariableOp.dense_60/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
§
E__inference_dense_57_layer_call_and_return_conditional_losses_1769903

inputs0
matmul_readvariableop_resource:7u-
biasadd_readvariableop_resource:u
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense_57/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7u*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿur
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:u*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
.dense_57/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7u*
dtype0
dense_57/kernel/Regularizer/AbsAbs6dense_57/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7ur
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum#dense_57/kernel/Regularizer/Abs:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *û9);
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu¨
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_57/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_57/kernel/Regularizer/Abs/ReadVariableOp.dense_57/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
¿°
1
"__inference__wrapped_model_1766796
normalization_6_input&
"sequential_6_normalization_6_sub_y'
#sequential_6_normalization_6_sqrt_xF
4sequential_6_dense_55_matmul_readvariableop_resource:7C
5sequential_6_dense_55_biasadd_readvariableop_resource:7S
Esequential_6_batch_normalization_49_batchnorm_readvariableop_resource:7W
Isequential_6_batch_normalization_49_batchnorm_mul_readvariableop_resource:7U
Gsequential_6_batch_normalization_49_batchnorm_readvariableop_1_resource:7U
Gsequential_6_batch_normalization_49_batchnorm_readvariableop_2_resource:7F
4sequential_6_dense_56_matmul_readvariableop_resource:77C
5sequential_6_dense_56_biasadd_readvariableop_resource:7S
Esequential_6_batch_normalization_50_batchnorm_readvariableop_resource:7W
Isequential_6_batch_normalization_50_batchnorm_mul_readvariableop_resource:7U
Gsequential_6_batch_normalization_50_batchnorm_readvariableop_1_resource:7U
Gsequential_6_batch_normalization_50_batchnorm_readvariableop_2_resource:7F
4sequential_6_dense_57_matmul_readvariableop_resource:7uC
5sequential_6_dense_57_biasadd_readvariableop_resource:uS
Esequential_6_batch_normalization_51_batchnorm_readvariableop_resource:uW
Isequential_6_batch_normalization_51_batchnorm_mul_readvariableop_resource:uU
Gsequential_6_batch_normalization_51_batchnorm_readvariableop_1_resource:uU
Gsequential_6_batch_normalization_51_batchnorm_readvariableop_2_resource:uF
4sequential_6_dense_58_matmul_readvariableop_resource:uC
5sequential_6_dense_58_biasadd_readvariableop_resource:S
Esequential_6_batch_normalization_52_batchnorm_readvariableop_resource:W
Isequential_6_batch_normalization_52_batchnorm_mul_readvariableop_resource:U
Gsequential_6_batch_normalization_52_batchnorm_readvariableop_1_resource:U
Gsequential_6_batch_normalization_52_batchnorm_readvariableop_2_resource:F
4sequential_6_dense_59_matmul_readvariableop_resource:C
5sequential_6_dense_59_biasadd_readvariableop_resource:S
Esequential_6_batch_normalization_53_batchnorm_readvariableop_resource:W
Isequential_6_batch_normalization_53_batchnorm_mul_readvariableop_resource:U
Gsequential_6_batch_normalization_53_batchnorm_readvariableop_1_resource:U
Gsequential_6_batch_normalization_53_batchnorm_readvariableop_2_resource:F
4sequential_6_dense_60_matmul_readvariableop_resource:C
5sequential_6_dense_60_biasadd_readvariableop_resource:S
Esequential_6_batch_normalization_54_batchnorm_readvariableop_resource:W
Isequential_6_batch_normalization_54_batchnorm_mul_readvariableop_resource:U
Gsequential_6_batch_normalization_54_batchnorm_readvariableop_1_resource:U
Gsequential_6_batch_normalization_54_batchnorm_readvariableop_2_resource:F
4sequential_6_dense_61_matmul_readvariableop_resource:C
5sequential_6_dense_61_biasadd_readvariableop_resource:S
Esequential_6_batch_normalization_55_batchnorm_readvariableop_resource:W
Isequential_6_batch_normalization_55_batchnorm_mul_readvariableop_resource:U
Gsequential_6_batch_normalization_55_batchnorm_readvariableop_1_resource:U
Gsequential_6_batch_normalization_55_batchnorm_readvariableop_2_resource:F
4sequential_6_dense_62_matmul_readvariableop_resource:C
5sequential_6_dense_62_biasadd_readvariableop_resource:
identity¢<sequential_6/batch_normalization_49/batchnorm/ReadVariableOp¢>sequential_6/batch_normalization_49/batchnorm/ReadVariableOp_1¢>sequential_6/batch_normalization_49/batchnorm/ReadVariableOp_2¢@sequential_6/batch_normalization_49/batchnorm/mul/ReadVariableOp¢<sequential_6/batch_normalization_50/batchnorm/ReadVariableOp¢>sequential_6/batch_normalization_50/batchnorm/ReadVariableOp_1¢>sequential_6/batch_normalization_50/batchnorm/ReadVariableOp_2¢@sequential_6/batch_normalization_50/batchnorm/mul/ReadVariableOp¢<sequential_6/batch_normalization_51/batchnorm/ReadVariableOp¢>sequential_6/batch_normalization_51/batchnorm/ReadVariableOp_1¢>sequential_6/batch_normalization_51/batchnorm/ReadVariableOp_2¢@sequential_6/batch_normalization_51/batchnorm/mul/ReadVariableOp¢<sequential_6/batch_normalization_52/batchnorm/ReadVariableOp¢>sequential_6/batch_normalization_52/batchnorm/ReadVariableOp_1¢>sequential_6/batch_normalization_52/batchnorm/ReadVariableOp_2¢@sequential_6/batch_normalization_52/batchnorm/mul/ReadVariableOp¢<sequential_6/batch_normalization_53/batchnorm/ReadVariableOp¢>sequential_6/batch_normalization_53/batchnorm/ReadVariableOp_1¢>sequential_6/batch_normalization_53/batchnorm/ReadVariableOp_2¢@sequential_6/batch_normalization_53/batchnorm/mul/ReadVariableOp¢<sequential_6/batch_normalization_54/batchnorm/ReadVariableOp¢>sequential_6/batch_normalization_54/batchnorm/ReadVariableOp_1¢>sequential_6/batch_normalization_54/batchnorm/ReadVariableOp_2¢@sequential_6/batch_normalization_54/batchnorm/mul/ReadVariableOp¢<sequential_6/batch_normalization_55/batchnorm/ReadVariableOp¢>sequential_6/batch_normalization_55/batchnorm/ReadVariableOp_1¢>sequential_6/batch_normalization_55/batchnorm/ReadVariableOp_2¢@sequential_6/batch_normalization_55/batchnorm/mul/ReadVariableOp¢,sequential_6/dense_55/BiasAdd/ReadVariableOp¢+sequential_6/dense_55/MatMul/ReadVariableOp¢,sequential_6/dense_56/BiasAdd/ReadVariableOp¢+sequential_6/dense_56/MatMul/ReadVariableOp¢,sequential_6/dense_57/BiasAdd/ReadVariableOp¢+sequential_6/dense_57/MatMul/ReadVariableOp¢,sequential_6/dense_58/BiasAdd/ReadVariableOp¢+sequential_6/dense_58/MatMul/ReadVariableOp¢,sequential_6/dense_59/BiasAdd/ReadVariableOp¢+sequential_6/dense_59/MatMul/ReadVariableOp¢,sequential_6/dense_60/BiasAdd/ReadVariableOp¢+sequential_6/dense_60/MatMul/ReadVariableOp¢,sequential_6/dense_61/BiasAdd/ReadVariableOp¢+sequential_6/dense_61/MatMul/ReadVariableOp¢,sequential_6/dense_62/BiasAdd/ReadVariableOp¢+sequential_6/dense_62/MatMul/ReadVariableOp
 sequential_6/normalization_6/subSubnormalization_6_input"sequential_6_normalization_6_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
!sequential_6/normalization_6/SqrtSqrt#sequential_6_normalization_6_sqrt_x*
T0*
_output_shapes

:k
&sequential_6/normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3°
$sequential_6/normalization_6/MaximumMaximum%sequential_6/normalization_6/Sqrt:y:0/sequential_6/normalization_6/Maximum/y:output:0*
T0*
_output_shapes

:±
$sequential_6/normalization_6/truedivRealDiv$sequential_6/normalization_6/sub:z:0(sequential_6/normalization_6/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+sequential_6/dense_55/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_55_matmul_readvariableop_resource*
_output_shapes

:7*
dtype0·
sequential_6/dense_55/MatMulMatMul(sequential_6/normalization_6/truediv:z:03sequential_6/dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
,sequential_6/dense_55/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_55_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0¸
sequential_6/dense_55/BiasAddBiasAdd&sequential_6/dense_55/MatMul:product:04sequential_6/dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¾
<sequential_6/batch_normalization_49/batchnorm/ReadVariableOpReadVariableOpEsequential_6_batch_normalization_49_batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0x
3sequential_6/batch_normalization_49/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ã
1sequential_6/batch_normalization_49/batchnorm/addAddV2Dsequential_6/batch_normalization_49/batchnorm/ReadVariableOp:value:0<sequential_6/batch_normalization_49/batchnorm/add/y:output:0*
T0*
_output_shapes
:7
3sequential_6/batch_normalization_49/batchnorm/RsqrtRsqrt5sequential_6/batch_normalization_49/batchnorm/add:z:0*
T0*
_output_shapes
:7Æ
@sequential_6/batch_normalization_49/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_6_batch_normalization_49_batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0à
1sequential_6/batch_normalization_49/batchnorm/mulMul7sequential_6/batch_normalization_49/batchnorm/Rsqrt:y:0Hsequential_6/batch_normalization_49/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7Ë
3sequential_6/batch_normalization_49/batchnorm/mul_1Mul&sequential_6/dense_55/BiasAdd:output:05sequential_6/batch_normalization_49/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7Â
>sequential_6/batch_normalization_49/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_6_batch_normalization_49_batchnorm_readvariableop_1_resource*
_output_shapes
:7*
dtype0Þ
3sequential_6/batch_normalization_49/batchnorm/mul_2MulFsequential_6/batch_normalization_49/batchnorm/ReadVariableOp_1:value:05sequential_6/batch_normalization_49/batchnorm/mul:z:0*
T0*
_output_shapes
:7Â
>sequential_6/batch_normalization_49/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_6_batch_normalization_49_batchnorm_readvariableop_2_resource*
_output_shapes
:7*
dtype0Þ
1sequential_6/batch_normalization_49/batchnorm/subSubFsequential_6/batch_normalization_49/batchnorm/ReadVariableOp_2:value:07sequential_6/batch_normalization_49/batchnorm/mul_2:z:0*
T0*
_output_shapes
:7Þ
3sequential_6/batch_normalization_49/batchnorm/add_1AddV27sequential_6/batch_normalization_49/batchnorm/mul_1:z:05sequential_6/batch_normalization_49/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¤
%sequential_6/leaky_re_lu_49/LeakyRelu	LeakyRelu7sequential_6/batch_normalization_49/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%> 
+sequential_6/dense_56/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_56_matmul_readvariableop_resource*
_output_shapes

:77*
dtype0Â
sequential_6/dense_56/MatMulMatMul3sequential_6/leaky_re_lu_49/LeakyRelu:activations:03sequential_6/dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
,sequential_6/dense_56/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_56_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0¸
sequential_6/dense_56/BiasAddBiasAdd&sequential_6/dense_56/MatMul:product:04sequential_6/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¾
<sequential_6/batch_normalization_50/batchnorm/ReadVariableOpReadVariableOpEsequential_6_batch_normalization_50_batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0x
3sequential_6/batch_normalization_50/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ã
1sequential_6/batch_normalization_50/batchnorm/addAddV2Dsequential_6/batch_normalization_50/batchnorm/ReadVariableOp:value:0<sequential_6/batch_normalization_50/batchnorm/add/y:output:0*
T0*
_output_shapes
:7
3sequential_6/batch_normalization_50/batchnorm/RsqrtRsqrt5sequential_6/batch_normalization_50/batchnorm/add:z:0*
T0*
_output_shapes
:7Æ
@sequential_6/batch_normalization_50/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_6_batch_normalization_50_batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0à
1sequential_6/batch_normalization_50/batchnorm/mulMul7sequential_6/batch_normalization_50/batchnorm/Rsqrt:y:0Hsequential_6/batch_normalization_50/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7Ë
3sequential_6/batch_normalization_50/batchnorm/mul_1Mul&sequential_6/dense_56/BiasAdd:output:05sequential_6/batch_normalization_50/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7Â
>sequential_6/batch_normalization_50/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_6_batch_normalization_50_batchnorm_readvariableop_1_resource*
_output_shapes
:7*
dtype0Þ
3sequential_6/batch_normalization_50/batchnorm/mul_2MulFsequential_6/batch_normalization_50/batchnorm/ReadVariableOp_1:value:05sequential_6/batch_normalization_50/batchnorm/mul:z:0*
T0*
_output_shapes
:7Â
>sequential_6/batch_normalization_50/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_6_batch_normalization_50_batchnorm_readvariableop_2_resource*
_output_shapes
:7*
dtype0Þ
1sequential_6/batch_normalization_50/batchnorm/subSubFsequential_6/batch_normalization_50/batchnorm/ReadVariableOp_2:value:07sequential_6/batch_normalization_50/batchnorm/mul_2:z:0*
T0*
_output_shapes
:7Þ
3sequential_6/batch_normalization_50/batchnorm/add_1AddV27sequential_6/batch_normalization_50/batchnorm/mul_1:z:05sequential_6/batch_normalization_50/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¤
%sequential_6/leaky_re_lu_50/LeakyRelu	LeakyRelu7sequential_6/batch_normalization_50/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%> 
+sequential_6/dense_57/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_57_matmul_readvariableop_resource*
_output_shapes

:7u*
dtype0Â
sequential_6/dense_57/MatMulMatMul3sequential_6/leaky_re_lu_50/LeakyRelu:activations:03sequential_6/dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
,sequential_6/dense_57/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_57_biasadd_readvariableop_resource*
_output_shapes
:u*
dtype0¸
sequential_6/dense_57/BiasAddBiasAdd&sequential_6/dense_57/MatMul:product:04sequential_6/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu¾
<sequential_6/batch_normalization_51/batchnorm/ReadVariableOpReadVariableOpEsequential_6_batch_normalization_51_batchnorm_readvariableop_resource*
_output_shapes
:u*
dtype0x
3sequential_6/batch_normalization_51/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ã
1sequential_6/batch_normalization_51/batchnorm/addAddV2Dsequential_6/batch_normalization_51/batchnorm/ReadVariableOp:value:0<sequential_6/batch_normalization_51/batchnorm/add/y:output:0*
T0*
_output_shapes
:u
3sequential_6/batch_normalization_51/batchnorm/RsqrtRsqrt5sequential_6/batch_normalization_51/batchnorm/add:z:0*
T0*
_output_shapes
:uÆ
@sequential_6/batch_normalization_51/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_6_batch_normalization_51_batchnorm_mul_readvariableop_resource*
_output_shapes
:u*
dtype0à
1sequential_6/batch_normalization_51/batchnorm/mulMul7sequential_6/batch_normalization_51/batchnorm/Rsqrt:y:0Hsequential_6/batch_normalization_51/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:uË
3sequential_6/batch_normalization_51/batchnorm/mul_1Mul&sequential_6/dense_57/BiasAdd:output:05sequential_6/batch_normalization_51/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿuÂ
>sequential_6/batch_normalization_51/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_6_batch_normalization_51_batchnorm_readvariableop_1_resource*
_output_shapes
:u*
dtype0Þ
3sequential_6/batch_normalization_51/batchnorm/mul_2MulFsequential_6/batch_normalization_51/batchnorm/ReadVariableOp_1:value:05sequential_6/batch_normalization_51/batchnorm/mul:z:0*
T0*
_output_shapes
:uÂ
>sequential_6/batch_normalization_51/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_6_batch_normalization_51_batchnorm_readvariableop_2_resource*
_output_shapes
:u*
dtype0Þ
1sequential_6/batch_normalization_51/batchnorm/subSubFsequential_6/batch_normalization_51/batchnorm/ReadVariableOp_2:value:07sequential_6/batch_normalization_51/batchnorm/mul_2:z:0*
T0*
_output_shapes
:uÞ
3sequential_6/batch_normalization_51/batchnorm/add_1AddV27sequential_6/batch_normalization_51/batchnorm/mul_1:z:05sequential_6/batch_normalization_51/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu¤
%sequential_6/leaky_re_lu_51/LeakyRelu	LeakyRelu7sequential_6/batch_normalization_51/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*
alpha%> 
+sequential_6/dense_58/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_58_matmul_readvariableop_resource*
_output_shapes

:u*
dtype0Â
sequential_6/dense_58/MatMulMatMul3sequential_6/leaky_re_lu_51/LeakyRelu:activations:03sequential_6/dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_6/dense_58/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¸
sequential_6/dense_58/BiasAddBiasAdd&sequential_6/dense_58/MatMul:product:04sequential_6/dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
<sequential_6/batch_normalization_52/batchnorm/ReadVariableOpReadVariableOpEsequential_6_batch_normalization_52_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0x
3sequential_6/batch_normalization_52/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ã
1sequential_6/batch_normalization_52/batchnorm/addAddV2Dsequential_6/batch_normalization_52/batchnorm/ReadVariableOp:value:0<sequential_6/batch_normalization_52/batchnorm/add/y:output:0*
T0*
_output_shapes
:
3sequential_6/batch_normalization_52/batchnorm/RsqrtRsqrt5sequential_6/batch_normalization_52/batchnorm/add:z:0*
T0*
_output_shapes
:Æ
@sequential_6/batch_normalization_52/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_6_batch_normalization_52_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0à
1sequential_6/batch_normalization_52/batchnorm/mulMul7sequential_6/batch_normalization_52/batchnorm/Rsqrt:y:0Hsequential_6/batch_normalization_52/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ë
3sequential_6/batch_normalization_52/batchnorm/mul_1Mul&sequential_6/dense_58/BiasAdd:output:05sequential_6/batch_normalization_52/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_6/batch_normalization_52/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_6_batch_normalization_52_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0Þ
3sequential_6/batch_normalization_52/batchnorm/mul_2MulFsequential_6/batch_normalization_52/batchnorm/ReadVariableOp_1:value:05sequential_6/batch_normalization_52/batchnorm/mul:z:0*
T0*
_output_shapes
:Â
>sequential_6/batch_normalization_52/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_6_batch_normalization_52_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0Þ
1sequential_6/batch_normalization_52/batchnorm/subSubFsequential_6/batch_normalization_52/batchnorm/ReadVariableOp_2:value:07sequential_6/batch_normalization_52/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Þ
3sequential_6/batch_normalization_52/batchnorm/add_1AddV27sequential_6/batch_normalization_52/batchnorm/mul_1:z:05sequential_6/batch_normalization_52/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
%sequential_6/leaky_re_lu_52/LeakyRelu	LeakyRelu7sequential_6/batch_normalization_52/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%> 
+sequential_6/dense_59/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_59_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Â
sequential_6/dense_59/MatMulMatMul3sequential_6/leaky_re_lu_52/LeakyRelu:activations:03sequential_6/dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_6/dense_59/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¸
sequential_6/dense_59/BiasAddBiasAdd&sequential_6/dense_59/MatMul:product:04sequential_6/dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
<sequential_6/batch_normalization_53/batchnorm/ReadVariableOpReadVariableOpEsequential_6_batch_normalization_53_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0x
3sequential_6/batch_normalization_53/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ã
1sequential_6/batch_normalization_53/batchnorm/addAddV2Dsequential_6/batch_normalization_53/batchnorm/ReadVariableOp:value:0<sequential_6/batch_normalization_53/batchnorm/add/y:output:0*
T0*
_output_shapes
:
3sequential_6/batch_normalization_53/batchnorm/RsqrtRsqrt5sequential_6/batch_normalization_53/batchnorm/add:z:0*
T0*
_output_shapes
:Æ
@sequential_6/batch_normalization_53/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_6_batch_normalization_53_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0à
1sequential_6/batch_normalization_53/batchnorm/mulMul7sequential_6/batch_normalization_53/batchnorm/Rsqrt:y:0Hsequential_6/batch_normalization_53/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ë
3sequential_6/batch_normalization_53/batchnorm/mul_1Mul&sequential_6/dense_59/BiasAdd:output:05sequential_6/batch_normalization_53/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_6/batch_normalization_53/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_6_batch_normalization_53_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0Þ
3sequential_6/batch_normalization_53/batchnorm/mul_2MulFsequential_6/batch_normalization_53/batchnorm/ReadVariableOp_1:value:05sequential_6/batch_normalization_53/batchnorm/mul:z:0*
T0*
_output_shapes
:Â
>sequential_6/batch_normalization_53/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_6_batch_normalization_53_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0Þ
1sequential_6/batch_normalization_53/batchnorm/subSubFsequential_6/batch_normalization_53/batchnorm/ReadVariableOp_2:value:07sequential_6/batch_normalization_53/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Þ
3sequential_6/batch_normalization_53/batchnorm/add_1AddV27sequential_6/batch_normalization_53/batchnorm/mul_1:z:05sequential_6/batch_normalization_53/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
%sequential_6/leaky_re_lu_53/LeakyRelu	LeakyRelu7sequential_6/batch_normalization_53/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%> 
+sequential_6/dense_60/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_60_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Â
sequential_6/dense_60/MatMulMatMul3sequential_6/leaky_re_lu_53/LeakyRelu:activations:03sequential_6/dense_60/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_6/dense_60/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¸
sequential_6/dense_60/BiasAddBiasAdd&sequential_6/dense_60/MatMul:product:04sequential_6/dense_60/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
<sequential_6/batch_normalization_54/batchnorm/ReadVariableOpReadVariableOpEsequential_6_batch_normalization_54_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0x
3sequential_6/batch_normalization_54/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ã
1sequential_6/batch_normalization_54/batchnorm/addAddV2Dsequential_6/batch_normalization_54/batchnorm/ReadVariableOp:value:0<sequential_6/batch_normalization_54/batchnorm/add/y:output:0*
T0*
_output_shapes
:
3sequential_6/batch_normalization_54/batchnorm/RsqrtRsqrt5sequential_6/batch_normalization_54/batchnorm/add:z:0*
T0*
_output_shapes
:Æ
@sequential_6/batch_normalization_54/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_6_batch_normalization_54_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0à
1sequential_6/batch_normalization_54/batchnorm/mulMul7sequential_6/batch_normalization_54/batchnorm/Rsqrt:y:0Hsequential_6/batch_normalization_54/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ë
3sequential_6/batch_normalization_54/batchnorm/mul_1Mul&sequential_6/dense_60/BiasAdd:output:05sequential_6/batch_normalization_54/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_6/batch_normalization_54/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_6_batch_normalization_54_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0Þ
3sequential_6/batch_normalization_54/batchnorm/mul_2MulFsequential_6/batch_normalization_54/batchnorm/ReadVariableOp_1:value:05sequential_6/batch_normalization_54/batchnorm/mul:z:0*
T0*
_output_shapes
:Â
>sequential_6/batch_normalization_54/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_6_batch_normalization_54_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0Þ
1sequential_6/batch_normalization_54/batchnorm/subSubFsequential_6/batch_normalization_54/batchnorm/ReadVariableOp_2:value:07sequential_6/batch_normalization_54/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Þ
3sequential_6/batch_normalization_54/batchnorm/add_1AddV27sequential_6/batch_normalization_54/batchnorm/mul_1:z:05sequential_6/batch_normalization_54/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
%sequential_6/leaky_re_lu_54/LeakyRelu	LeakyRelu7sequential_6/batch_normalization_54/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%> 
+sequential_6/dense_61/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_61_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Â
sequential_6/dense_61/MatMulMatMul3sequential_6/leaky_re_lu_54/LeakyRelu:activations:03sequential_6/dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_6/dense_61/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¸
sequential_6/dense_61/BiasAddBiasAdd&sequential_6/dense_61/MatMul:product:04sequential_6/dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
<sequential_6/batch_normalization_55/batchnorm/ReadVariableOpReadVariableOpEsequential_6_batch_normalization_55_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0x
3sequential_6/batch_normalization_55/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ã
1sequential_6/batch_normalization_55/batchnorm/addAddV2Dsequential_6/batch_normalization_55/batchnorm/ReadVariableOp:value:0<sequential_6/batch_normalization_55/batchnorm/add/y:output:0*
T0*
_output_shapes
:
3sequential_6/batch_normalization_55/batchnorm/RsqrtRsqrt5sequential_6/batch_normalization_55/batchnorm/add:z:0*
T0*
_output_shapes
:Æ
@sequential_6/batch_normalization_55/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_6_batch_normalization_55_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0à
1sequential_6/batch_normalization_55/batchnorm/mulMul7sequential_6/batch_normalization_55/batchnorm/Rsqrt:y:0Hsequential_6/batch_normalization_55/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ë
3sequential_6/batch_normalization_55/batchnorm/mul_1Mul&sequential_6/dense_61/BiasAdd:output:05sequential_6/batch_normalization_55/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_6/batch_normalization_55/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_6_batch_normalization_55_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0Þ
3sequential_6/batch_normalization_55/batchnorm/mul_2MulFsequential_6/batch_normalization_55/batchnorm/ReadVariableOp_1:value:05sequential_6/batch_normalization_55/batchnorm/mul:z:0*
T0*
_output_shapes
:Â
>sequential_6/batch_normalization_55/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_6_batch_normalization_55_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0Þ
1sequential_6/batch_normalization_55/batchnorm/subSubFsequential_6/batch_normalization_55/batchnorm/ReadVariableOp_2:value:07sequential_6/batch_normalization_55/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Þ
3sequential_6/batch_normalization_55/batchnorm/add_1AddV27sequential_6/batch_normalization_55/batchnorm/mul_1:z:05sequential_6/batch_normalization_55/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
%sequential_6/leaky_re_lu_55/LeakyRelu	LeakyRelu7sequential_6/batch_normalization_55/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%> 
+sequential_6/dense_62/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_62_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Â
sequential_6/dense_62/MatMulMatMul3sequential_6/leaky_re_lu_55/LeakyRelu:activations:03sequential_6/dense_62/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_6/dense_62/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¸
sequential_6/dense_62/BiasAddBiasAdd&sequential_6/dense_62/MatMul:product:04sequential_6/dense_62/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
IdentityIdentity&sequential_6/dense_62/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
NoOpNoOp=^sequential_6/batch_normalization_49/batchnorm/ReadVariableOp?^sequential_6/batch_normalization_49/batchnorm/ReadVariableOp_1?^sequential_6/batch_normalization_49/batchnorm/ReadVariableOp_2A^sequential_6/batch_normalization_49/batchnorm/mul/ReadVariableOp=^sequential_6/batch_normalization_50/batchnorm/ReadVariableOp?^sequential_6/batch_normalization_50/batchnorm/ReadVariableOp_1?^sequential_6/batch_normalization_50/batchnorm/ReadVariableOp_2A^sequential_6/batch_normalization_50/batchnorm/mul/ReadVariableOp=^sequential_6/batch_normalization_51/batchnorm/ReadVariableOp?^sequential_6/batch_normalization_51/batchnorm/ReadVariableOp_1?^sequential_6/batch_normalization_51/batchnorm/ReadVariableOp_2A^sequential_6/batch_normalization_51/batchnorm/mul/ReadVariableOp=^sequential_6/batch_normalization_52/batchnorm/ReadVariableOp?^sequential_6/batch_normalization_52/batchnorm/ReadVariableOp_1?^sequential_6/batch_normalization_52/batchnorm/ReadVariableOp_2A^sequential_6/batch_normalization_52/batchnorm/mul/ReadVariableOp=^sequential_6/batch_normalization_53/batchnorm/ReadVariableOp?^sequential_6/batch_normalization_53/batchnorm/ReadVariableOp_1?^sequential_6/batch_normalization_53/batchnorm/ReadVariableOp_2A^sequential_6/batch_normalization_53/batchnorm/mul/ReadVariableOp=^sequential_6/batch_normalization_54/batchnorm/ReadVariableOp?^sequential_6/batch_normalization_54/batchnorm/ReadVariableOp_1?^sequential_6/batch_normalization_54/batchnorm/ReadVariableOp_2A^sequential_6/batch_normalization_54/batchnorm/mul/ReadVariableOp=^sequential_6/batch_normalization_55/batchnorm/ReadVariableOp?^sequential_6/batch_normalization_55/batchnorm/ReadVariableOp_1?^sequential_6/batch_normalization_55/batchnorm/ReadVariableOp_2A^sequential_6/batch_normalization_55/batchnorm/mul/ReadVariableOp-^sequential_6/dense_55/BiasAdd/ReadVariableOp,^sequential_6/dense_55/MatMul/ReadVariableOp-^sequential_6/dense_56/BiasAdd/ReadVariableOp,^sequential_6/dense_56/MatMul/ReadVariableOp-^sequential_6/dense_57/BiasAdd/ReadVariableOp,^sequential_6/dense_57/MatMul/ReadVariableOp-^sequential_6/dense_58/BiasAdd/ReadVariableOp,^sequential_6/dense_58/MatMul/ReadVariableOp-^sequential_6/dense_59/BiasAdd/ReadVariableOp,^sequential_6/dense_59/MatMul/ReadVariableOp-^sequential_6/dense_60/BiasAdd/ReadVariableOp,^sequential_6/dense_60/MatMul/ReadVariableOp-^sequential_6/dense_61/BiasAdd/ReadVariableOp,^sequential_6/dense_61/MatMul/ReadVariableOp-^sequential_6/dense_62/BiasAdd/ReadVariableOp,^sequential_6/dense_62/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2|
<sequential_6/batch_normalization_49/batchnorm/ReadVariableOp<sequential_6/batch_normalization_49/batchnorm/ReadVariableOp2
>sequential_6/batch_normalization_49/batchnorm/ReadVariableOp_1>sequential_6/batch_normalization_49/batchnorm/ReadVariableOp_12
>sequential_6/batch_normalization_49/batchnorm/ReadVariableOp_2>sequential_6/batch_normalization_49/batchnorm/ReadVariableOp_22
@sequential_6/batch_normalization_49/batchnorm/mul/ReadVariableOp@sequential_6/batch_normalization_49/batchnorm/mul/ReadVariableOp2|
<sequential_6/batch_normalization_50/batchnorm/ReadVariableOp<sequential_6/batch_normalization_50/batchnorm/ReadVariableOp2
>sequential_6/batch_normalization_50/batchnorm/ReadVariableOp_1>sequential_6/batch_normalization_50/batchnorm/ReadVariableOp_12
>sequential_6/batch_normalization_50/batchnorm/ReadVariableOp_2>sequential_6/batch_normalization_50/batchnorm/ReadVariableOp_22
@sequential_6/batch_normalization_50/batchnorm/mul/ReadVariableOp@sequential_6/batch_normalization_50/batchnorm/mul/ReadVariableOp2|
<sequential_6/batch_normalization_51/batchnorm/ReadVariableOp<sequential_6/batch_normalization_51/batchnorm/ReadVariableOp2
>sequential_6/batch_normalization_51/batchnorm/ReadVariableOp_1>sequential_6/batch_normalization_51/batchnorm/ReadVariableOp_12
>sequential_6/batch_normalization_51/batchnorm/ReadVariableOp_2>sequential_6/batch_normalization_51/batchnorm/ReadVariableOp_22
@sequential_6/batch_normalization_51/batchnorm/mul/ReadVariableOp@sequential_6/batch_normalization_51/batchnorm/mul/ReadVariableOp2|
<sequential_6/batch_normalization_52/batchnorm/ReadVariableOp<sequential_6/batch_normalization_52/batchnorm/ReadVariableOp2
>sequential_6/batch_normalization_52/batchnorm/ReadVariableOp_1>sequential_6/batch_normalization_52/batchnorm/ReadVariableOp_12
>sequential_6/batch_normalization_52/batchnorm/ReadVariableOp_2>sequential_6/batch_normalization_52/batchnorm/ReadVariableOp_22
@sequential_6/batch_normalization_52/batchnorm/mul/ReadVariableOp@sequential_6/batch_normalization_52/batchnorm/mul/ReadVariableOp2|
<sequential_6/batch_normalization_53/batchnorm/ReadVariableOp<sequential_6/batch_normalization_53/batchnorm/ReadVariableOp2
>sequential_6/batch_normalization_53/batchnorm/ReadVariableOp_1>sequential_6/batch_normalization_53/batchnorm/ReadVariableOp_12
>sequential_6/batch_normalization_53/batchnorm/ReadVariableOp_2>sequential_6/batch_normalization_53/batchnorm/ReadVariableOp_22
@sequential_6/batch_normalization_53/batchnorm/mul/ReadVariableOp@sequential_6/batch_normalization_53/batchnorm/mul/ReadVariableOp2|
<sequential_6/batch_normalization_54/batchnorm/ReadVariableOp<sequential_6/batch_normalization_54/batchnorm/ReadVariableOp2
>sequential_6/batch_normalization_54/batchnorm/ReadVariableOp_1>sequential_6/batch_normalization_54/batchnorm/ReadVariableOp_12
>sequential_6/batch_normalization_54/batchnorm/ReadVariableOp_2>sequential_6/batch_normalization_54/batchnorm/ReadVariableOp_22
@sequential_6/batch_normalization_54/batchnorm/mul/ReadVariableOp@sequential_6/batch_normalization_54/batchnorm/mul/ReadVariableOp2|
<sequential_6/batch_normalization_55/batchnorm/ReadVariableOp<sequential_6/batch_normalization_55/batchnorm/ReadVariableOp2
>sequential_6/batch_normalization_55/batchnorm/ReadVariableOp_1>sequential_6/batch_normalization_55/batchnorm/ReadVariableOp_12
>sequential_6/batch_normalization_55/batchnorm/ReadVariableOp_2>sequential_6/batch_normalization_55/batchnorm/ReadVariableOp_22
@sequential_6/batch_normalization_55/batchnorm/mul/ReadVariableOp@sequential_6/batch_normalization_55/batchnorm/mul/ReadVariableOp2\
,sequential_6/dense_55/BiasAdd/ReadVariableOp,sequential_6/dense_55/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_55/MatMul/ReadVariableOp+sequential_6/dense_55/MatMul/ReadVariableOp2\
,sequential_6/dense_56/BiasAdd/ReadVariableOp,sequential_6/dense_56/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_56/MatMul/ReadVariableOp+sequential_6/dense_56/MatMul/ReadVariableOp2\
,sequential_6/dense_57/BiasAdd/ReadVariableOp,sequential_6/dense_57/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_57/MatMul/ReadVariableOp+sequential_6/dense_57/MatMul/ReadVariableOp2\
,sequential_6/dense_58/BiasAdd/ReadVariableOp,sequential_6/dense_58/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_58/MatMul/ReadVariableOp+sequential_6/dense_58/MatMul/ReadVariableOp2\
,sequential_6/dense_59/BiasAdd/ReadVariableOp,sequential_6/dense_59/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_59/MatMul/ReadVariableOp+sequential_6/dense_59/MatMul/ReadVariableOp2\
,sequential_6/dense_60/BiasAdd/ReadVariableOp,sequential_6/dense_60/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_60/MatMul/ReadVariableOp+sequential_6/dense_60/MatMul/ReadVariableOp2\
,sequential_6/dense_61/BiasAdd/ReadVariableOp,sequential_6/dense_61/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_61/MatMul/ReadVariableOp+sequential_6/dense_61/MatMul/ReadVariableOp2\
,sequential_6/dense_62/BiasAdd/ReadVariableOp,sequential_6/dense_62/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_62/MatMul/ReadVariableOp+sequential_6/dense_62/MatMul/ReadVariableOp:^ Z
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namenormalization_6_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ð
²
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1767312

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
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
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1769862

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
Ó
8__inference_batch_normalization_49_layer_call_fn_1769674

inputs
unknown:7
	unknown_0:7
	unknown_1:7
	unknown_2:7
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1766820o
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
Ð
²
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1766902

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
«
L
0__inference_leaky_re_lu_53_layer_call_fn_1770230

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_1767572`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
L
0__inference_leaky_re_lu_50_layer_call_fn_1769867

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
:ÿÿÿÿÿÿÿÿÿ7* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_1767458`
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
«
L
0__inference_leaky_re_lu_52_layer_call_fn_1770109

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_1767534`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1767230

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
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
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1767113

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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
Ä

*__inference_dense_60_layer_call_fn_1770250

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_60_layer_call_and_return_conditional_losses_1767590o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö


.__inference_sequential_6_layer_call_fn_1768849

inputs
unknown
	unknown_0
	unknown_1:7
	unknown_2:7
	unknown_3:7
	unknown_4:7
	unknown_5:7
	unknown_6:7
	unknown_7:77
	unknown_8:7
	unknown_9:7

unknown_10:7

unknown_11:7

unknown_12:7

unknown_13:7u

unknown_14:u

unknown_15:u

unknown_16:u

unknown_17:u

unknown_18:u

unknown_19:u

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:
identity¢StatefulPartitionedCall»
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,-.*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_1767709o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
å
g
K__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_1770477

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È


.__inference_sequential_6_layer_call_fn_1768946

inputs
unknown
	unknown_0
	unknown_1:7
	unknown_2:7
	unknown_3:7
	unknown_4:7
	unknown_5:7
	unknown_6:7
	unknown_7:77
	unknown_8:7
	unknown_9:7

unknown_10:7

unknown_11:7

unknown_12:7

unknown_13:7u

unknown_14:u

unknown_15:u

unknown_16:u

unknown_17:u

unknown_18:u

unknown_19:u

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:
identity¢StatefulPartitionedCall­
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
unknown_44*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*@
_read_only_resource_inputs"
 	
 !"%&'(+,-.*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_6_layer_call_and_return_conditional_losses_1768188o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
%
ì
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1769741

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
¾
§
E__inference_dense_56_layer_call_and_return_conditional_losses_1767438

inputs0
matmul_readvariableop_resource:77-
biasadd_readvariableop_resource:7
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense_56/kernel/Regularizer/Abs/ReadVariableOpt
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
:ÿÿÿÿÿÿÿÿÿ7
.dense_56/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:77*
dtype0
dense_56/kernel/Regularizer/AbsAbs6dense_56/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77r
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum#dense_56/kernel/Regularizer/Abs:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¨
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_56/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_56/kernel/Regularizer/Abs/ReadVariableOp.dense_56/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_52_layer_call_fn_1770050

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1767113o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

I__inference_sequential_6_layer_call_and_return_conditional_losses_1768706
normalization_6_input
normalization_6_sub_y
normalization_6_sqrt_x"
dense_55_1768553:7
dense_55_1768555:7,
batch_normalization_49_1768558:7,
batch_normalization_49_1768560:7,
batch_normalization_49_1768562:7,
batch_normalization_49_1768564:7"
dense_56_1768568:77
dense_56_1768570:7,
batch_normalization_50_1768573:7,
batch_normalization_50_1768575:7,
batch_normalization_50_1768577:7,
batch_normalization_50_1768579:7"
dense_57_1768583:7u
dense_57_1768585:u,
batch_normalization_51_1768588:u,
batch_normalization_51_1768590:u,
batch_normalization_51_1768592:u,
batch_normalization_51_1768594:u"
dense_58_1768598:u
dense_58_1768600:,
batch_normalization_52_1768603:,
batch_normalization_52_1768605:,
batch_normalization_52_1768607:,
batch_normalization_52_1768609:"
dense_59_1768613:
dense_59_1768615:,
batch_normalization_53_1768618:,
batch_normalization_53_1768620:,
batch_normalization_53_1768622:,
batch_normalization_53_1768624:"
dense_60_1768628:
dense_60_1768630:,
batch_normalization_54_1768633:,
batch_normalization_54_1768635:,
batch_normalization_54_1768637:,
batch_normalization_54_1768639:"
dense_61_1768643:
dense_61_1768645:,
batch_normalization_55_1768648:,
batch_normalization_55_1768650:,
batch_normalization_55_1768652:,
batch_normalization_55_1768654:"
dense_62_1768658:
dense_62_1768660:
identity¢.batch_normalization_49/StatefulPartitionedCall¢.batch_normalization_50/StatefulPartitionedCall¢.batch_normalization_51/StatefulPartitionedCall¢.batch_normalization_52/StatefulPartitionedCall¢.batch_normalization_53/StatefulPartitionedCall¢.batch_normalization_54/StatefulPartitionedCall¢.batch_normalization_55/StatefulPartitionedCall¢ dense_55/StatefulPartitionedCall¢.dense_55/kernel/Regularizer/Abs/ReadVariableOp¢ dense_56/StatefulPartitionedCall¢.dense_56/kernel/Regularizer/Abs/ReadVariableOp¢ dense_57/StatefulPartitionedCall¢.dense_57/kernel/Regularizer/Abs/ReadVariableOp¢ dense_58/StatefulPartitionedCall¢.dense_58/kernel/Regularizer/Abs/ReadVariableOp¢ dense_59/StatefulPartitionedCall¢.dense_59/kernel/Regularizer/Abs/ReadVariableOp¢ dense_60/StatefulPartitionedCall¢.dense_60/kernel/Regularizer/Abs/ReadVariableOp¢ dense_61/StatefulPartitionedCall¢.dense_61/kernel/Regularizer/Abs/ReadVariableOp¢ dense_62/StatefulPartitionedCallz
normalization_6/subSubnormalization_6_inputnormalization_6_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
normalization_6/SqrtSqrtnormalization_6_sqrt_x*
T0*
_output_shapes

:^
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_55/StatefulPartitionedCallStatefulPartitionedCallnormalization_6/truediv:z:0dense_55_1768553dense_55_1768555*
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
GPU 2J 8 *N
fIRG
E__inference_dense_55_layer_call_and_return_conditional_losses_1767400
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0batch_normalization_49_1768558batch_normalization_49_1768560batch_normalization_49_1768562batch_normalization_49_1768564*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1766867ö
leaky_re_lu_49/PartitionedCallPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_1767420
 dense_56/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_49/PartitionedCall:output:0dense_56_1768568dense_56_1768570*
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
GPU 2J 8 *N
fIRG
E__inference_dense_56_layer_call_and_return_conditional_losses_1767438
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0batch_normalization_50_1768573batch_normalization_50_1768575batch_normalization_50_1768577batch_normalization_50_1768579*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1766949ö
leaky_re_lu_50/PartitionedCallPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_1767458
 dense_57/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_50/PartitionedCall:output:0dense_57_1768583dense_57_1768585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_57_layer_call_and_return_conditional_losses_1767476
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0batch_normalization_51_1768588batch_normalization_51_1768590batch_normalization_51_1768592batch_normalization_51_1768594*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1767031ö
leaky_re_lu_51/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_1767496
 dense_58/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_51/PartitionedCall:output:0dense_58_1768598dense_58_1768600*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_58_layer_call_and_return_conditional_losses_1767514
.batch_normalization_52/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0batch_normalization_52_1768603batch_normalization_52_1768605batch_normalization_52_1768607batch_normalization_52_1768609*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1767113ö
leaky_re_lu_52/PartitionedCallPartitionedCall7batch_normalization_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_1767534
 dense_59/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_52/PartitionedCall:output:0dense_59_1768613dense_59_1768615*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_1767552
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall)dense_59/StatefulPartitionedCall:output:0batch_normalization_53_1768618batch_normalization_53_1768620batch_normalization_53_1768622batch_normalization_53_1768624*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1767195ö
leaky_re_lu_53/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_1767572
 dense_60/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_53/PartitionedCall:output:0dense_60_1768628dense_60_1768630*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_60_layer_call_and_return_conditional_losses_1767590
.batch_normalization_54/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0batch_normalization_54_1768633batch_normalization_54_1768635batch_normalization_54_1768637batch_normalization_54_1768639*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1767277ö
leaky_re_lu_54/PartitionedCallPartitionedCall7batch_normalization_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_1767610
 dense_61/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_54/PartitionedCall:output:0dense_61_1768643dense_61_1768645*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_61_layer_call_and_return_conditional_losses_1767628
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0batch_normalization_55_1768648batch_normalization_55_1768650batch_normalization_55_1768652batch_normalization_55_1768654*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1767359ö
leaky_re_lu_55/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_1767648
 dense_62/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_55/PartitionedCall:output:0dense_62_1768658dense_62_1768660*
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
E__inference_dense_62_layer_call_and_return_conditional_losses_1767660
.dense_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_55_1768553*
_output_shapes

:7*
dtype0
dense_55/kernel/Regularizer/AbsAbs6dense_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7r
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_55/kernel/Regularizer/SumSum#dense_55/kernel/Regularizer/Abs:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_56/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_56_1768568*
_output_shapes

:77*
dtype0
dense_56/kernel/Regularizer/AbsAbs6dense_56/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77r
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum#dense_56/kernel/Regularizer/Abs:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_57/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_57_1768583*
_output_shapes

:7u*
dtype0
dense_57/kernel/Regularizer/AbsAbs6dense_57/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7ur
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum#dense_57/kernel/Regularizer/Abs:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *û9);
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_58/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_58_1768598*
_output_shapes

:u*
dtype0
dense_58/kernel/Regularizer/AbsAbs6dense_58/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:ur
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum#dense_58/kernel/Regularizer/Abs:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_59/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_59_1768613*
_output_shapes

:*
dtype0
dense_59/kernel/Regularizer/AbsAbs6dense_59/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum#dense_59/kernel/Regularizer/Abs:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_60/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_60_1768628*
_output_shapes

:*
dtype0
dense_60/kernel/Regularizer/AbsAbs6dense_60/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_60/kernel/Regularizer/SumSum#dense_60/kernel/Regularizer/Abs:y:0*dense_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_60/kernel/Regularizer/mulMul*dense_60/kernel/Regularizer/mul/x:output:0(dense_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_61/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_61_1768643*
_output_shapes

:*
dtype0
dense_61/kernel/Regularizer/AbsAbs6dense_61/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_61/kernel/Regularizer/SumSum#dense_61/kernel/Regularizer/Abs:y:0*dense_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_61/kernel/Regularizer/mulMul*dense_61/kernel/Regularizer/mul/x:output:0(dense_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_62/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_52/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_54/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall/^dense_55/kernel/Regularizer/Abs/ReadVariableOp!^dense_56/StatefulPartitionedCall/^dense_56/kernel/Regularizer/Abs/ReadVariableOp!^dense_57/StatefulPartitionedCall/^dense_57/kernel/Regularizer/Abs/ReadVariableOp!^dense_58/StatefulPartitionedCall/^dense_58/kernel/Regularizer/Abs/ReadVariableOp!^dense_59/StatefulPartitionedCall/^dense_59/kernel/Regularizer/Abs/ReadVariableOp!^dense_60/StatefulPartitionedCall/^dense_60/kernel/Regularizer/Abs/ReadVariableOp!^dense_61/StatefulPartitionedCall/^dense_61/kernel/Regularizer/Abs/ReadVariableOp!^dense_62/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_52/StatefulPartitionedCall.batch_normalization_52/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_54/StatefulPartitionedCall.batch_normalization_54/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2`
.dense_55/kernel/Regularizer/Abs/ReadVariableOp.dense_55/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2`
.dense_56/kernel/Regularizer/Abs/ReadVariableOp.dense_56/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2`
.dense_57/kernel/Regularizer/Abs/ReadVariableOp.dense_57/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2`
.dense_58/kernel/Regularizer/Abs/ReadVariableOp.dense_58/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2`
.dense_59/kernel/Regularizer/Abs/ReadVariableOp.dense_59/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2`
.dense_60/kernel/Regularizer/Abs/ReadVariableOp.dense_60/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2`
.dense_61/kernel/Regularizer/Abs/ReadVariableOp.dense_61/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall:^ Z
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namenormalization_6_input:$ 

_output_shapes

::$ 

_output_shapes

:
ï'
Ó
__inference_adapt_step_1769630
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
Ð
²
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1770433

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
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
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1770346

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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
%
ì
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1767195

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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
¾
§
E__inference_dense_59_layer_call_and_return_conditional_losses_1770145

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense_59/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.dense_59/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_59/kernel/Regularizer/AbsAbs6dense_59/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum#dense_59/kernel/Regularizer/Abs:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_59/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_59/kernel/Regularizer/Abs/ReadVariableOp.dense_59/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¬
__inference_loss_fn_6_1770573I
7dense_61_kernel_regularizer_abs_readvariableop_resource:
identity¢.dense_61/kernel/Regularizer/Abs/ReadVariableOp¦
.dense_61/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_61_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:*
dtype0
dense_61/kernel/Regularizer/AbsAbs6dense_61/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_61/kernel/Regularizer/SumSum#dense_61/kernel/Regularizer/Abs:y:0*dense_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_61/kernel/Regularizer/mulMul*dense_61/kernel/Regularizer/mul/x:output:0(dense_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_61/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^dense_61/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_61/kernel/Regularizer/Abs/ReadVariableOp.dense_61/kernel/Regularizer/Abs/ReadVariableOp
å
g
K__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_1769872

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
Ð
²
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1769828

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
å
g
K__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_1767420

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
%
ì
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1769983

inputs5
'assignmovingavg_readvariableop_resource:u7
)assignmovingavg_1_readvariableop_resource:u3
%batchnorm_mul_readvariableop_resource:u/
!batchnorm_readvariableop_resource:u
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:u*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:u
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿul
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:u*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:u*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:u*
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
:u*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:ux
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:u¬
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
:u*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:u~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:u´
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
:uP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:u~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:u*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:uc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿuh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:uv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:u*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ur
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿub
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿuê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿu: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_51_layer_call_fn_1769929

inputs
unknown:u
	unknown_0:u
	unknown_1:u
	unknown_2:u
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1767031o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿu: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 
_user_specified_nameinputs
â­

I__inference_sequential_6_layer_call_and_return_conditional_losses_1767709

inputs
normalization_6_sub_y
normalization_6_sqrt_x"
dense_55_1767401:7
dense_55_1767403:7,
batch_normalization_49_1767406:7,
batch_normalization_49_1767408:7,
batch_normalization_49_1767410:7,
batch_normalization_49_1767412:7"
dense_56_1767439:77
dense_56_1767441:7,
batch_normalization_50_1767444:7,
batch_normalization_50_1767446:7,
batch_normalization_50_1767448:7,
batch_normalization_50_1767450:7"
dense_57_1767477:7u
dense_57_1767479:u,
batch_normalization_51_1767482:u,
batch_normalization_51_1767484:u,
batch_normalization_51_1767486:u,
batch_normalization_51_1767488:u"
dense_58_1767515:u
dense_58_1767517:,
batch_normalization_52_1767520:,
batch_normalization_52_1767522:,
batch_normalization_52_1767524:,
batch_normalization_52_1767526:"
dense_59_1767553:
dense_59_1767555:,
batch_normalization_53_1767558:,
batch_normalization_53_1767560:,
batch_normalization_53_1767562:,
batch_normalization_53_1767564:"
dense_60_1767591:
dense_60_1767593:,
batch_normalization_54_1767596:,
batch_normalization_54_1767598:,
batch_normalization_54_1767600:,
batch_normalization_54_1767602:"
dense_61_1767629:
dense_61_1767631:,
batch_normalization_55_1767634:,
batch_normalization_55_1767636:,
batch_normalization_55_1767638:,
batch_normalization_55_1767640:"
dense_62_1767661:
dense_62_1767663:
identity¢.batch_normalization_49/StatefulPartitionedCall¢.batch_normalization_50/StatefulPartitionedCall¢.batch_normalization_51/StatefulPartitionedCall¢.batch_normalization_52/StatefulPartitionedCall¢.batch_normalization_53/StatefulPartitionedCall¢.batch_normalization_54/StatefulPartitionedCall¢.batch_normalization_55/StatefulPartitionedCall¢ dense_55/StatefulPartitionedCall¢.dense_55/kernel/Regularizer/Abs/ReadVariableOp¢ dense_56/StatefulPartitionedCall¢.dense_56/kernel/Regularizer/Abs/ReadVariableOp¢ dense_57/StatefulPartitionedCall¢.dense_57/kernel/Regularizer/Abs/ReadVariableOp¢ dense_58/StatefulPartitionedCall¢.dense_58/kernel/Regularizer/Abs/ReadVariableOp¢ dense_59/StatefulPartitionedCall¢.dense_59/kernel/Regularizer/Abs/ReadVariableOp¢ dense_60/StatefulPartitionedCall¢.dense_60/kernel/Regularizer/Abs/ReadVariableOp¢ dense_61/StatefulPartitionedCall¢.dense_61/kernel/Regularizer/Abs/ReadVariableOp¢ dense_62/StatefulPartitionedCallk
normalization_6/subSubinputsnormalization_6_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
normalization_6/SqrtSqrtnormalization_6_sqrt_x*
T0*
_output_shapes

:^
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_55/StatefulPartitionedCallStatefulPartitionedCallnormalization_6/truediv:z:0dense_55_1767401dense_55_1767403*
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
GPU 2J 8 *N
fIRG
E__inference_dense_55_layer_call_and_return_conditional_losses_1767400
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0batch_normalization_49_1767406batch_normalization_49_1767408batch_normalization_49_1767410batch_normalization_49_1767412*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1766820ö
leaky_re_lu_49/PartitionedCallPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_1767420
 dense_56/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_49/PartitionedCall:output:0dense_56_1767439dense_56_1767441*
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
GPU 2J 8 *N
fIRG
E__inference_dense_56_layer_call_and_return_conditional_losses_1767438
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0batch_normalization_50_1767444batch_normalization_50_1767446batch_normalization_50_1767448batch_normalization_50_1767450*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1766902ö
leaky_re_lu_50/PartitionedCallPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_1767458
 dense_57/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_50/PartitionedCall:output:0dense_57_1767477dense_57_1767479*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_57_layer_call_and_return_conditional_losses_1767476
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0batch_normalization_51_1767482batch_normalization_51_1767484batch_normalization_51_1767486batch_normalization_51_1767488*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1766984ö
leaky_re_lu_51/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_1767496
 dense_58/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_51/PartitionedCall:output:0dense_58_1767515dense_58_1767517*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_58_layer_call_and_return_conditional_losses_1767514
.batch_normalization_52/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0batch_normalization_52_1767520batch_normalization_52_1767522batch_normalization_52_1767524batch_normalization_52_1767526*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1767066ö
leaky_re_lu_52/PartitionedCallPartitionedCall7batch_normalization_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_1767534
 dense_59/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_52/PartitionedCall:output:0dense_59_1767553dense_59_1767555*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_1767552
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall)dense_59/StatefulPartitionedCall:output:0batch_normalization_53_1767558batch_normalization_53_1767560batch_normalization_53_1767562batch_normalization_53_1767564*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1767148ö
leaky_re_lu_53/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_1767572
 dense_60/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_53/PartitionedCall:output:0dense_60_1767591dense_60_1767593*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_60_layer_call_and_return_conditional_losses_1767590
.batch_normalization_54/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0batch_normalization_54_1767596batch_normalization_54_1767598batch_normalization_54_1767600batch_normalization_54_1767602*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1767230ö
leaky_re_lu_54/PartitionedCallPartitionedCall7batch_normalization_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_1767610
 dense_61/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_54/PartitionedCall:output:0dense_61_1767629dense_61_1767631*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_61_layer_call_and_return_conditional_losses_1767628
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0batch_normalization_55_1767634batch_normalization_55_1767636batch_normalization_55_1767638batch_normalization_55_1767640*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1767312ö
leaky_re_lu_55/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_1767648
 dense_62/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_55/PartitionedCall:output:0dense_62_1767661dense_62_1767663*
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
E__inference_dense_62_layer_call_and_return_conditional_losses_1767660
.dense_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_55_1767401*
_output_shapes

:7*
dtype0
dense_55/kernel/Regularizer/AbsAbs6dense_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7r
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_55/kernel/Regularizer/SumSum#dense_55/kernel/Regularizer/Abs:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_56/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_56_1767439*
_output_shapes

:77*
dtype0
dense_56/kernel/Regularizer/AbsAbs6dense_56/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77r
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum#dense_56/kernel/Regularizer/Abs:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_57/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_57_1767477*
_output_shapes

:7u*
dtype0
dense_57/kernel/Regularizer/AbsAbs6dense_57/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7ur
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum#dense_57/kernel/Regularizer/Abs:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *û9);
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_58/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_58_1767515*
_output_shapes

:u*
dtype0
dense_58/kernel/Regularizer/AbsAbs6dense_58/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:ur
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum#dense_58/kernel/Regularizer/Abs:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_59/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_59_1767553*
_output_shapes

:*
dtype0
dense_59/kernel/Regularizer/AbsAbs6dense_59/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum#dense_59/kernel/Regularizer/Abs:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_60/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_60_1767591*
_output_shapes

:*
dtype0
dense_60/kernel/Regularizer/AbsAbs6dense_60/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_60/kernel/Regularizer/SumSum#dense_60/kernel/Regularizer/Abs:y:0*dense_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_60/kernel/Regularizer/mulMul*dense_60/kernel/Regularizer/mul/x:output:0(dense_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_61/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_61_1767629*
_output_shapes

:*
dtype0
dense_61/kernel/Regularizer/AbsAbs6dense_61/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_61/kernel/Regularizer/SumSum#dense_61/kernel/Regularizer/Abs:y:0*dense_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_61/kernel/Regularizer/mulMul*dense_61/kernel/Regularizer/mul/x:output:0(dense_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_62/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_52/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_54/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall/^dense_55/kernel/Regularizer/Abs/ReadVariableOp!^dense_56/StatefulPartitionedCall/^dense_56/kernel/Regularizer/Abs/ReadVariableOp!^dense_57/StatefulPartitionedCall/^dense_57/kernel/Regularizer/Abs/ReadVariableOp!^dense_58/StatefulPartitionedCall/^dense_58/kernel/Regularizer/Abs/ReadVariableOp!^dense_59/StatefulPartitionedCall/^dense_59/kernel/Regularizer/Abs/ReadVariableOp!^dense_60/StatefulPartitionedCall/^dense_60/kernel/Regularizer/Abs/ReadVariableOp!^dense_61/StatefulPartitionedCall/^dense_61/kernel/Regularizer/Abs/ReadVariableOp!^dense_62/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_52/StatefulPartitionedCall.batch_normalization_52/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_54/StatefulPartitionedCall.batch_normalization_54/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2`
.dense_55/kernel/Regularizer/Abs/ReadVariableOp.dense_55/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2`
.dense_56/kernel/Regularizer/Abs/ReadVariableOp.dense_56/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2`
.dense_57/kernel/Regularizer/Abs/ReadVariableOp.dense_57/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2`
.dense_58/kernel/Regularizer/Abs/ReadVariableOp.dense_58/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2`
.dense_59/kernel/Regularizer/Abs/ReadVariableOp.dense_59/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2`
.dense_60/kernel/Regularizer/Abs/ReadVariableOp.dense_60/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2`
.dense_61/kernel/Regularizer/Abs/ReadVariableOp.dense_61/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
%
ì
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1767277

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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

¬
__inference_loss_fn_4_1770551I
7dense_59_kernel_regularizer_abs_readvariableop_resource:
identity¢.dense_59/kernel/Regularizer/Abs/ReadVariableOp¦
.dense_59/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_59_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:*
dtype0
dense_59/kernel/Regularizer/AbsAbs6dense_59/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum#dense_59/kernel/Regularizer/Abs:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_59/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^dense_59/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_59/kernel/Regularizer/Abs/ReadVariableOp.dense_59/kernel/Regularizer/Abs/ReadVariableOp
¬
Ó
8__inference_batch_normalization_54_layer_call_fn_1770279

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1767230o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
Ó
8__inference_batch_normalization_50_layer_call_fn_1769808

inputs
unknown:7
	unknown_0:7
	unknown_1:7
	unknown_2:7
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1766949o
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

¬
__inference_loss_fn_1_1770518I
7dense_56_kernel_regularizer_abs_readvariableop_resource:77
identity¢.dense_56/kernel/Regularizer/Abs/ReadVariableOp¦
.dense_56/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_56_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:77*
dtype0
dense_56/kernel/Regularizer/AbsAbs6dense_56/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77r
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum#dense_56/kernel/Regularizer/Abs:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_56/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^dense_56/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_56/kernel/Regularizer/Abs/ReadVariableOp.dense_56/kernel/Regularizer/Abs/ReadVariableOp
»Ù
4
 __inference__traced_save_1770937
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	.
*savev2_dense_55_kernel_read_readvariableop,
(savev2_dense_55_bias_read_readvariableop;
7savev2_batch_normalization_49_gamma_read_readvariableop:
6savev2_batch_normalization_49_beta_read_readvariableopA
=savev2_batch_normalization_49_moving_mean_read_readvariableopE
Asavev2_batch_normalization_49_moving_variance_read_readvariableop.
*savev2_dense_56_kernel_read_readvariableop,
(savev2_dense_56_bias_read_readvariableop;
7savev2_batch_normalization_50_gamma_read_readvariableop:
6savev2_batch_normalization_50_beta_read_readvariableopA
=savev2_batch_normalization_50_moving_mean_read_readvariableopE
Asavev2_batch_normalization_50_moving_variance_read_readvariableop.
*savev2_dense_57_kernel_read_readvariableop,
(savev2_dense_57_bias_read_readvariableop;
7savev2_batch_normalization_51_gamma_read_readvariableop:
6savev2_batch_normalization_51_beta_read_readvariableopA
=savev2_batch_normalization_51_moving_mean_read_readvariableopE
Asavev2_batch_normalization_51_moving_variance_read_readvariableop.
*savev2_dense_58_kernel_read_readvariableop,
(savev2_dense_58_bias_read_readvariableop;
7savev2_batch_normalization_52_gamma_read_readvariableop:
6savev2_batch_normalization_52_beta_read_readvariableopA
=savev2_batch_normalization_52_moving_mean_read_readvariableopE
Asavev2_batch_normalization_52_moving_variance_read_readvariableop.
*savev2_dense_59_kernel_read_readvariableop,
(savev2_dense_59_bias_read_readvariableop;
7savev2_batch_normalization_53_gamma_read_readvariableop:
6savev2_batch_normalization_53_beta_read_readvariableopA
=savev2_batch_normalization_53_moving_mean_read_readvariableopE
Asavev2_batch_normalization_53_moving_variance_read_readvariableop.
*savev2_dense_60_kernel_read_readvariableop,
(savev2_dense_60_bias_read_readvariableop;
7savev2_batch_normalization_54_gamma_read_readvariableop:
6savev2_batch_normalization_54_beta_read_readvariableopA
=savev2_batch_normalization_54_moving_mean_read_readvariableopE
Asavev2_batch_normalization_54_moving_variance_read_readvariableop.
*savev2_dense_61_kernel_read_readvariableop,
(savev2_dense_61_bias_read_readvariableop;
7savev2_batch_normalization_55_gamma_read_readvariableop:
6savev2_batch_normalization_55_beta_read_readvariableopA
=savev2_batch_normalization_55_moving_mean_read_readvariableopE
Asavev2_batch_normalization_55_moving_variance_read_readvariableop.
*savev2_dense_62_kernel_read_readvariableop,
(savev2_dense_62_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_55_kernel_m_read_readvariableop3
/savev2_adam_dense_55_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_49_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_49_beta_m_read_readvariableop5
1savev2_adam_dense_56_kernel_m_read_readvariableop3
/savev2_adam_dense_56_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_50_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_50_beta_m_read_readvariableop5
1savev2_adam_dense_57_kernel_m_read_readvariableop3
/savev2_adam_dense_57_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_51_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_51_beta_m_read_readvariableop5
1savev2_adam_dense_58_kernel_m_read_readvariableop3
/savev2_adam_dense_58_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_52_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_52_beta_m_read_readvariableop5
1savev2_adam_dense_59_kernel_m_read_readvariableop3
/savev2_adam_dense_59_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_53_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_53_beta_m_read_readvariableop5
1savev2_adam_dense_60_kernel_m_read_readvariableop3
/savev2_adam_dense_60_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_54_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_54_beta_m_read_readvariableop5
1savev2_adam_dense_61_kernel_m_read_readvariableop3
/savev2_adam_dense_61_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_55_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_55_beta_m_read_readvariableop5
1savev2_adam_dense_62_kernel_m_read_readvariableop3
/savev2_adam_dense_62_bias_m_read_readvariableop5
1savev2_adam_dense_55_kernel_v_read_readvariableop3
/savev2_adam_dense_55_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_49_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_49_beta_v_read_readvariableop5
1savev2_adam_dense_56_kernel_v_read_readvariableop3
/savev2_adam_dense_56_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_50_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_50_beta_v_read_readvariableop5
1savev2_adam_dense_57_kernel_v_read_readvariableop3
/savev2_adam_dense_57_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_51_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_51_beta_v_read_readvariableop5
1savev2_adam_dense_58_kernel_v_read_readvariableop3
/savev2_adam_dense_58_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_52_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_52_beta_v_read_readvariableop5
1savev2_adam_dense_59_kernel_v_read_readvariableop3
/savev2_adam_dense_59_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_53_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_53_beta_v_read_readvariableop5
1savev2_adam_dense_60_kernel_v_read_readvariableop3
/savev2_adam_dense_60_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_54_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_54_beta_v_read_readvariableop5
1savev2_adam_dense_61_kernel_v_read_readvariableop3
/savev2_adam_dense_61_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_55_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_55_beta_v_read_readvariableop5
1savev2_adam_dense_62_kernel_v_read_readvariableop3
/savev2_adam_dense_62_bias_v_read_readvariableop
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
: »?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:r*
dtype0*ä>
valueÚ>B×>rB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÔ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:r*
dtype0*ù
valueïBìrB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ö1
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop*savev2_dense_55_kernel_read_readvariableop(savev2_dense_55_bias_read_readvariableop7savev2_batch_normalization_49_gamma_read_readvariableop6savev2_batch_normalization_49_beta_read_readvariableop=savev2_batch_normalization_49_moving_mean_read_readvariableopAsavev2_batch_normalization_49_moving_variance_read_readvariableop*savev2_dense_56_kernel_read_readvariableop(savev2_dense_56_bias_read_readvariableop7savev2_batch_normalization_50_gamma_read_readvariableop6savev2_batch_normalization_50_beta_read_readvariableop=savev2_batch_normalization_50_moving_mean_read_readvariableopAsavev2_batch_normalization_50_moving_variance_read_readvariableop*savev2_dense_57_kernel_read_readvariableop(savev2_dense_57_bias_read_readvariableop7savev2_batch_normalization_51_gamma_read_readvariableop6savev2_batch_normalization_51_beta_read_readvariableop=savev2_batch_normalization_51_moving_mean_read_readvariableopAsavev2_batch_normalization_51_moving_variance_read_readvariableop*savev2_dense_58_kernel_read_readvariableop(savev2_dense_58_bias_read_readvariableop7savev2_batch_normalization_52_gamma_read_readvariableop6savev2_batch_normalization_52_beta_read_readvariableop=savev2_batch_normalization_52_moving_mean_read_readvariableopAsavev2_batch_normalization_52_moving_variance_read_readvariableop*savev2_dense_59_kernel_read_readvariableop(savev2_dense_59_bias_read_readvariableop7savev2_batch_normalization_53_gamma_read_readvariableop6savev2_batch_normalization_53_beta_read_readvariableop=savev2_batch_normalization_53_moving_mean_read_readvariableopAsavev2_batch_normalization_53_moving_variance_read_readvariableop*savev2_dense_60_kernel_read_readvariableop(savev2_dense_60_bias_read_readvariableop7savev2_batch_normalization_54_gamma_read_readvariableop6savev2_batch_normalization_54_beta_read_readvariableop=savev2_batch_normalization_54_moving_mean_read_readvariableopAsavev2_batch_normalization_54_moving_variance_read_readvariableop*savev2_dense_61_kernel_read_readvariableop(savev2_dense_61_bias_read_readvariableop7savev2_batch_normalization_55_gamma_read_readvariableop6savev2_batch_normalization_55_beta_read_readvariableop=savev2_batch_normalization_55_moving_mean_read_readvariableopAsavev2_batch_normalization_55_moving_variance_read_readvariableop*savev2_dense_62_kernel_read_readvariableop(savev2_dense_62_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_55_kernel_m_read_readvariableop/savev2_adam_dense_55_bias_m_read_readvariableop>savev2_adam_batch_normalization_49_gamma_m_read_readvariableop=savev2_adam_batch_normalization_49_beta_m_read_readvariableop1savev2_adam_dense_56_kernel_m_read_readvariableop/savev2_adam_dense_56_bias_m_read_readvariableop>savev2_adam_batch_normalization_50_gamma_m_read_readvariableop=savev2_adam_batch_normalization_50_beta_m_read_readvariableop1savev2_adam_dense_57_kernel_m_read_readvariableop/savev2_adam_dense_57_bias_m_read_readvariableop>savev2_adam_batch_normalization_51_gamma_m_read_readvariableop=savev2_adam_batch_normalization_51_beta_m_read_readvariableop1savev2_adam_dense_58_kernel_m_read_readvariableop/savev2_adam_dense_58_bias_m_read_readvariableop>savev2_adam_batch_normalization_52_gamma_m_read_readvariableop=savev2_adam_batch_normalization_52_beta_m_read_readvariableop1savev2_adam_dense_59_kernel_m_read_readvariableop/savev2_adam_dense_59_bias_m_read_readvariableop>savev2_adam_batch_normalization_53_gamma_m_read_readvariableop=savev2_adam_batch_normalization_53_beta_m_read_readvariableop1savev2_adam_dense_60_kernel_m_read_readvariableop/savev2_adam_dense_60_bias_m_read_readvariableop>savev2_adam_batch_normalization_54_gamma_m_read_readvariableop=savev2_adam_batch_normalization_54_beta_m_read_readvariableop1savev2_adam_dense_61_kernel_m_read_readvariableop/savev2_adam_dense_61_bias_m_read_readvariableop>savev2_adam_batch_normalization_55_gamma_m_read_readvariableop=savev2_adam_batch_normalization_55_beta_m_read_readvariableop1savev2_adam_dense_62_kernel_m_read_readvariableop/savev2_adam_dense_62_bias_m_read_readvariableop1savev2_adam_dense_55_kernel_v_read_readvariableop/savev2_adam_dense_55_bias_v_read_readvariableop>savev2_adam_batch_normalization_49_gamma_v_read_readvariableop=savev2_adam_batch_normalization_49_beta_v_read_readvariableop1savev2_adam_dense_56_kernel_v_read_readvariableop/savev2_adam_dense_56_bias_v_read_readvariableop>savev2_adam_batch_normalization_50_gamma_v_read_readvariableop=savev2_adam_batch_normalization_50_beta_v_read_readvariableop1savev2_adam_dense_57_kernel_v_read_readvariableop/savev2_adam_dense_57_bias_v_read_readvariableop>savev2_adam_batch_normalization_51_gamma_v_read_readvariableop=savev2_adam_batch_normalization_51_beta_v_read_readvariableop1savev2_adam_dense_58_kernel_v_read_readvariableop/savev2_adam_dense_58_bias_v_read_readvariableop>savev2_adam_batch_normalization_52_gamma_v_read_readvariableop=savev2_adam_batch_normalization_52_beta_v_read_readvariableop1savev2_adam_dense_59_kernel_v_read_readvariableop/savev2_adam_dense_59_bias_v_read_readvariableop>savev2_adam_batch_normalization_53_gamma_v_read_readvariableop=savev2_adam_batch_normalization_53_beta_v_read_readvariableop1savev2_adam_dense_60_kernel_v_read_readvariableop/savev2_adam_dense_60_bias_v_read_readvariableop>savev2_adam_batch_normalization_54_gamma_v_read_readvariableop=savev2_adam_batch_normalization_54_beta_v_read_readvariableop1savev2_adam_dense_61_kernel_v_read_readvariableop/savev2_adam_dense_61_bias_v_read_readvariableop>savev2_adam_batch_normalization_55_gamma_v_read_readvariableop=savev2_adam_batch_normalization_55_beta_v_read_readvariableop1savev2_adam_dense_62_kernel_v_read_readvariableop/savev2_adam_dense_62_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *
dtypesv
t2r		
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

identity_1Identity_1:output:0*
_input_shapesñ
î: ::: :7:7:7:7:7:7:77:7:7:7:7:7:7u:u:u:u:u:u:u:::::::::::::::::::::::::: : : : : : :7:7:7:7:77:7:7:7:7u:u:u:u:u::::::::::::::::::7:7:7:7:77:7:7:7:7u:u:u:u:u:::::::::::::::::: 2(
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

:7: 

_output_shapes
:7: 

_output_shapes
:7: 

_output_shapes
:7: 

_output_shapes
:7: 	

_output_shapes
:7:$
 

_output_shapes

:77: 

_output_shapes
:7: 

_output_shapes
:7: 

_output_shapes
:7: 

_output_shapes
:7: 

_output_shapes
:7:$ 

_output_shapes

:7u: 

_output_shapes
:u: 

_output_shapes
:u: 

_output_shapes
:u: 

_output_shapes
:u: 

_output_shapes
:u:$ 

_output_shapes

:u: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
::$. 

_output_shapes

:: /

_output_shapes
::0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :$6 

_output_shapes

:7: 7
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

:77: ;

_output_shapes
:7: <

_output_shapes
:7: =

_output_shapes
:7:$> 

_output_shapes

:7u: ?

_output_shapes
:u: @

_output_shapes
:u: A

_output_shapes
:u:$B 

_output_shapes

:u: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
::$F 

_output_shapes

:: G

_output_shapes
:: H

_output_shapes
:: I

_output_shapes
::$J 

_output_shapes

:: K

_output_shapes
:: L

_output_shapes
:: M

_output_shapes
::$N 

_output_shapes

:: O

_output_shapes
:: P

_output_shapes
:: Q

_output_shapes
::$R 

_output_shapes

:: S

_output_shapes
::$T 

_output_shapes

:7: U

_output_shapes
:7: V

_output_shapes
:7: W

_output_shapes
:7:$X 

_output_shapes

:77: Y

_output_shapes
:7: Z

_output_shapes
:7: [

_output_shapes
:7:$\ 

_output_shapes

:7u: ]

_output_shapes
:u: ^

_output_shapes
:u: _

_output_shapes
:u:$` 

_output_shapes

:u: a

_output_shapes
:: b

_output_shapes
:: c

_output_shapes
::$d 

_output_shapes

:: e

_output_shapes
:: f

_output_shapes
:: g

_output_shapes
::$h 

_output_shapes

:: i

_output_shapes
:: j

_output_shapes
:: k

_output_shapes
::$l 

_output_shapes

:: m

_output_shapes
:: n

_output_shapes
:: o

_output_shapes
::$p 

_output_shapes

:: q

_output_shapes
::r

_output_shapes
: 
Ð
²
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1769949

inputs/
!batchnorm_readvariableop_resource:u3
%batchnorm_mul_readvariableop_resource:u1
#batchnorm_readvariableop_1_resource:u1
#batchnorm_readvariableop_2_resource:u
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:u*
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
:uP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:u~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:u*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:uc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿuz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:u*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:uz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:u*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:ur
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿub
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿuº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿu: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_52_layer_call_fn_1770037

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1767066o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1769707

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
®

I__inference_sequential_6_layer_call_and_return_conditional_losses_1768543
normalization_6_input
normalization_6_sub_y
normalization_6_sqrt_x"
dense_55_1768390:7
dense_55_1768392:7,
batch_normalization_49_1768395:7,
batch_normalization_49_1768397:7,
batch_normalization_49_1768399:7,
batch_normalization_49_1768401:7"
dense_56_1768405:77
dense_56_1768407:7,
batch_normalization_50_1768410:7,
batch_normalization_50_1768412:7,
batch_normalization_50_1768414:7,
batch_normalization_50_1768416:7"
dense_57_1768420:7u
dense_57_1768422:u,
batch_normalization_51_1768425:u,
batch_normalization_51_1768427:u,
batch_normalization_51_1768429:u,
batch_normalization_51_1768431:u"
dense_58_1768435:u
dense_58_1768437:,
batch_normalization_52_1768440:,
batch_normalization_52_1768442:,
batch_normalization_52_1768444:,
batch_normalization_52_1768446:"
dense_59_1768450:
dense_59_1768452:,
batch_normalization_53_1768455:,
batch_normalization_53_1768457:,
batch_normalization_53_1768459:,
batch_normalization_53_1768461:"
dense_60_1768465:
dense_60_1768467:,
batch_normalization_54_1768470:,
batch_normalization_54_1768472:,
batch_normalization_54_1768474:,
batch_normalization_54_1768476:"
dense_61_1768480:
dense_61_1768482:,
batch_normalization_55_1768485:,
batch_normalization_55_1768487:,
batch_normalization_55_1768489:,
batch_normalization_55_1768491:"
dense_62_1768495:
dense_62_1768497:
identity¢.batch_normalization_49/StatefulPartitionedCall¢.batch_normalization_50/StatefulPartitionedCall¢.batch_normalization_51/StatefulPartitionedCall¢.batch_normalization_52/StatefulPartitionedCall¢.batch_normalization_53/StatefulPartitionedCall¢.batch_normalization_54/StatefulPartitionedCall¢.batch_normalization_55/StatefulPartitionedCall¢ dense_55/StatefulPartitionedCall¢.dense_55/kernel/Regularizer/Abs/ReadVariableOp¢ dense_56/StatefulPartitionedCall¢.dense_56/kernel/Regularizer/Abs/ReadVariableOp¢ dense_57/StatefulPartitionedCall¢.dense_57/kernel/Regularizer/Abs/ReadVariableOp¢ dense_58/StatefulPartitionedCall¢.dense_58/kernel/Regularizer/Abs/ReadVariableOp¢ dense_59/StatefulPartitionedCall¢.dense_59/kernel/Regularizer/Abs/ReadVariableOp¢ dense_60/StatefulPartitionedCall¢.dense_60/kernel/Regularizer/Abs/ReadVariableOp¢ dense_61/StatefulPartitionedCall¢.dense_61/kernel/Regularizer/Abs/ReadVariableOp¢ dense_62/StatefulPartitionedCallz
normalization_6/subSubnormalization_6_inputnormalization_6_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
normalization_6/SqrtSqrtnormalization_6_sqrt_x*
T0*
_output_shapes

:^
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_55/StatefulPartitionedCallStatefulPartitionedCallnormalization_6/truediv:z:0dense_55_1768390dense_55_1768392*
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
GPU 2J 8 *N
fIRG
E__inference_dense_55_layer_call_and_return_conditional_losses_1767400
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0batch_normalization_49_1768395batch_normalization_49_1768397batch_normalization_49_1768399batch_normalization_49_1768401*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1766820ö
leaky_re_lu_49/PartitionedCallPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_1767420
 dense_56/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_49/PartitionedCall:output:0dense_56_1768405dense_56_1768407*
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
GPU 2J 8 *N
fIRG
E__inference_dense_56_layer_call_and_return_conditional_losses_1767438
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0batch_normalization_50_1768410batch_normalization_50_1768412batch_normalization_50_1768414batch_normalization_50_1768416*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1766902ö
leaky_re_lu_50/PartitionedCallPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_1767458
 dense_57/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_50/PartitionedCall:output:0dense_57_1768420dense_57_1768422*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_57_layer_call_and_return_conditional_losses_1767476
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0batch_normalization_51_1768425batch_normalization_51_1768427batch_normalization_51_1768429batch_normalization_51_1768431*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1766984ö
leaky_re_lu_51/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_1767496
 dense_58/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_51/PartitionedCall:output:0dense_58_1768435dense_58_1768437*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_58_layer_call_and_return_conditional_losses_1767514
.batch_normalization_52/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0batch_normalization_52_1768440batch_normalization_52_1768442batch_normalization_52_1768444batch_normalization_52_1768446*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1767066ö
leaky_re_lu_52/PartitionedCallPartitionedCall7batch_normalization_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_1767534
 dense_59/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_52/PartitionedCall:output:0dense_59_1768450dense_59_1768452*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_59_layer_call_and_return_conditional_losses_1767552
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall)dense_59/StatefulPartitionedCall:output:0batch_normalization_53_1768455batch_normalization_53_1768457batch_normalization_53_1768459batch_normalization_53_1768461*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1767148ö
leaky_re_lu_53/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_1767572
 dense_60/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_53/PartitionedCall:output:0dense_60_1768465dense_60_1768467*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_60_layer_call_and_return_conditional_losses_1767590
.batch_normalization_54/StatefulPartitionedCallStatefulPartitionedCall)dense_60/StatefulPartitionedCall:output:0batch_normalization_54_1768470batch_normalization_54_1768472batch_normalization_54_1768474batch_normalization_54_1768476*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1767230ö
leaky_re_lu_54/PartitionedCallPartitionedCall7batch_normalization_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_1767610
 dense_61/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_54/PartitionedCall:output:0dense_61_1768480dense_61_1768482*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_61_layer_call_and_return_conditional_losses_1767628
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall)dense_61/StatefulPartitionedCall:output:0batch_normalization_55_1768485batch_normalization_55_1768487batch_normalization_55_1768489batch_normalization_55_1768491*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1767312ö
leaky_re_lu_55/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_1767648
 dense_62/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_55/PartitionedCall:output:0dense_62_1768495dense_62_1768497*
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
E__inference_dense_62_layer_call_and_return_conditional_losses_1767660
.dense_55/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_55_1768390*
_output_shapes

:7*
dtype0
dense_55/kernel/Regularizer/AbsAbs6dense_55/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7r
!dense_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_55/kernel/Regularizer/SumSum#dense_55/kernel/Regularizer/Abs:y:0*dense_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_55/kernel/Regularizer/mulMul*dense_55/kernel/Regularizer/mul/x:output:0(dense_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_56/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_56_1768405*
_output_shapes

:77*
dtype0
dense_56/kernel/Regularizer/AbsAbs6dense_56/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:77r
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum#dense_56/kernel/Regularizer/Abs:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¯<
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_57/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_57_1768420*
_output_shapes

:7u*
dtype0
dense_57/kernel/Regularizer/AbsAbs6dense_57/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7ur
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum#dense_57/kernel/Regularizer/Abs:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *û9);
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_58/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_58_1768435*
_output_shapes

:u*
dtype0
dense_58/kernel/Regularizer/AbsAbs6dense_58/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:ur
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum#dense_58/kernel/Regularizer/Abs:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_59/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_59_1768450*
_output_shapes

:*
dtype0
dense_59/kernel/Regularizer/AbsAbs6dense_59/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum#dense_59/kernel/Regularizer/Abs:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_60/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_60_1768465*
_output_shapes

:*
dtype0
dense_60/kernel/Regularizer/AbsAbs6dense_60/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_60/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_60/kernel/Regularizer/SumSum#dense_60/kernel/Regularizer/Abs:y:0*dense_60/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_60/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_60/kernel/Regularizer/mulMul*dense_60/kernel/Regularizer/mul/x:output:0(dense_60/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
.dense_61/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_61_1768480*
_output_shapes

:*
dtype0
dense_61/kernel/Regularizer/AbsAbs6dense_61/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:r
!dense_61/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_61/kernel/Regularizer/SumSum#dense_61/kernel/Regularizer/Abs:y:0*dense_61/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_61/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *¹Ò<
dense_61/kernel/Regularizer/mulMul*dense_61/kernel/Regularizer/mul/x:output:0(dense_61/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_62/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_52/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_54/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall/^dense_55/kernel/Regularizer/Abs/ReadVariableOp!^dense_56/StatefulPartitionedCall/^dense_56/kernel/Regularizer/Abs/ReadVariableOp!^dense_57/StatefulPartitionedCall/^dense_57/kernel/Regularizer/Abs/ReadVariableOp!^dense_58/StatefulPartitionedCall/^dense_58/kernel/Regularizer/Abs/ReadVariableOp!^dense_59/StatefulPartitionedCall/^dense_59/kernel/Regularizer/Abs/ReadVariableOp!^dense_60/StatefulPartitionedCall/^dense_60/kernel/Regularizer/Abs/ReadVariableOp!^dense_61/StatefulPartitionedCall/^dense_61/kernel/Regularizer/Abs/ReadVariableOp!^dense_62/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_52/StatefulPartitionedCall.batch_normalization_52/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_54/StatefulPartitionedCall.batch_normalization_54/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2`
.dense_55/kernel/Regularizer/Abs/ReadVariableOp.dense_55/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2`
.dense_56/kernel/Regularizer/Abs/ReadVariableOp.dense_56/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2`
.dense_57/kernel/Regularizer/Abs/ReadVariableOp.dense_57/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2`
.dense_58/kernel/Regularizer/Abs/ReadVariableOp.dense_58/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2`
.dense_59/kernel/Regularizer/Abs/ReadVariableOp.dense_59/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_60/StatefulPartitionedCall dense_60/StatefulPartitionedCall2`
.dense_60/kernel/Regularizer/Abs/ReadVariableOp.dense_60/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2`
.dense_61/kernel/Regularizer/Abs/ReadVariableOp.dense_61/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_62/StatefulPartitionedCall dense_62/StatefulPartitionedCall:^ Z
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_namenormalization_6_input:$ 

_output_shapes

::$ 

_output_shapes

:
¾
§
E__inference_dense_57_layer_call_and_return_conditional_losses_1767476

inputs0
matmul_readvariableop_resource:7u-
biasadd_readvariableop_resource:u
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢.dense_57/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7u*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿur
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:u*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
.dense_57/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7u*
dtype0
dense_57/kernel/Regularizer/AbsAbs6dense_57/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7ur
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum#dense_57/kernel/Regularizer/Abs:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *û9);
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu¨
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_57/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_57/kernel/Regularizer/Abs/ReadVariableOp.dense_57/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
¬
Ó
8__inference_batch_normalization_50_layer_call_fn_1769795

inputs
unknown:7
	unknown_0:7
	unknown_1:7
	unknown_2:7
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1766902o
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
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ç
serving_default³
W
normalization_6_input>
'serving_default_normalization_6_input:0ÿÿÿÿÿÿÿÿÿ<
dense_620
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¡
Ä
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
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
 
signatures"
_tf_keras_sequential
Ó
!
_keep_axis
"_reduce_axis
#_reduce_axis_mask
$_broadcast_shape
%mean
%
adapt_mean
&variance
&adapt_variance
	'count
(	keras_api
)_adapt_function"
_tf_keras_layer
»

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
2axis
	3gamma
4beta
5moving_mean
6moving_variance
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Ckernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
»

\kernel
]bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
»

ukernel
vbias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
ò
}axis
	~gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
«
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
§kernel
	¨bias
©	variables
ªtrainable_variables
«regularization_losses
¬	keras_api
­__call__
+®&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	¯axis

°gamma
	±beta
²moving_mean
³moving_variance
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses"
_tf_keras_layer
«
º	variables
»trainable_variables
¼regularization_losses
½	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Àkernel
	Ábias
Â	variables
Ãtrainable_variables
Äregularization_losses
Å	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	Èaxis

Égamma
	Êbeta
Ëmoving_mean
Ìmoving_variance
Í	variables
Îtrainable_variables
Ïregularization_losses
Ð	keras_api
Ñ__call__
+Ò&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ó	variables
Ôtrainable_variables
Õregularization_losses
Ö	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ùkernel
	Úbias
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses"
_tf_keras_layer
¸
	áiter
âbeta_1
ãbeta_2

ädecay*må+mæ3mç4mèCméDmêLmëMmì\mí]mîemïfmðumñvmò~mómô	mõ	mö	m÷	mø	§mù	¨mú	°mû	±mü	Àmý	Ámþ	Émÿ	Êm	Ùm	Úm*v+v3v4vCvDvLvMv\v]vevfvuvvv~vv	v	v	v	v	§v	¨v	°v	±v	Àv	Áv	Év	Êv	Ùv	Úv "
	optimizer
¤
%0
&1
'2
*3
+4
35
46
57
68
C9
D10
L11
M12
N13
O14
\15
]16
e17
f18
g19
h20
u21
v22
~23
24
25
26
27
28
29
30
31
32
§33
¨34
°35
±36
²37
³38
À39
Á40
É41
Ê42
Ë43
Ì44
Ù45
Ú46"
trackable_list_wrapper

*0
+1
32
43
C4
D5
L6
M7
\8
]9
e10
f11
u12
v13
~14
15
16
17
18
19
§20
¨21
°22
±23
À24
Á25
É26
Ê27
Ù28
Ú29"
trackable_list_wrapper
X
å0
æ1
ç2
è3
é4
ê5
ë6"
trackable_list_wrapper
Ï
ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_sequential_6_layer_call_fn_1767804
.__inference_sequential_6_layer_call_fn_1768849
.__inference_sequential_6_layer_call_fn_1768946
.__inference_sequential_6_layer_call_fn_1768380À
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
I__inference_sequential_6_layer_call_and_return_conditional_losses_1769166
I__inference_sequential_6_layer_call_and_return_conditional_losses_1769484
I__inference_sequential_6_layer_call_and_return_conditional_losses_1768543
I__inference_sequential_6_layer_call_and_return_conditional_losses_1768706À
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
"__inference__wrapped_model_1766796normalization_6_input"
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
ñserving_default"
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
__inference_adapt_step_1769630
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
!:72dense_55/kernel
:72dense_55/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
(
å0"
trackable_list_wrapper
²
ònon_trainable_variables
ólayers
ômetrics
 õlayer_regularization_losses
ölayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_55_layer_call_fn_1769645¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
E__inference_dense_55_layer_call_and_return_conditional_losses_1769661¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
*:(72batch_normalization_49/gamma
):'72batch_normalization_49/beta
2:07 (2"batch_normalization_49/moving_mean
6:47 (2&batch_normalization_49/moving_variance
<
30
41
52
63"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
²
÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_49_layer_call_fn_1769674
8__inference_batch_normalization_49_layer_call_fn_1769687´
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
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1769707
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1769741´
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
ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_49_layer_call_fn_1769746¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
K__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_1769751¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:772dense_56/kernel
:72dense_56/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
(
æ0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_56_layer_call_fn_1769766¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
E__inference_dense_56_layer_call_and_return_conditional_losses_1769782¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
*:(72batch_normalization_50/gamma
):'72batch_normalization_50/beta
2:07 (2"batch_normalization_50/moving_mean
6:47 (2&batch_normalization_50/moving_variance
<
L0
M1
N2
O3"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_50_layer_call_fn_1769795
8__inference_batch_normalization_50_layer_call_fn_1769808´
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
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1769828
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1769862´
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_50_layer_call_fn_1769867¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
K__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_1769872¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:7u2dense_57/kernel
:u2dense_57/bias
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
(
ç0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_57_layer_call_fn_1769887¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
E__inference_dense_57_layer_call_and_return_conditional_losses_1769903¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
*:(u2batch_normalization_51/gamma
):'u2batch_normalization_51/beta
2:0u (2"batch_normalization_51/moving_mean
6:4u (2&batch_normalization_51/moving_variance
<
e0
f1
g2
h3"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_51_layer_call_fn_1769916
8__inference_batch_normalization_51_layer_call_fn_1769929´
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
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1769949
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1769983´
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_51_layer_call_fn_1769988¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
K__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_1769993¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:u2dense_58/kernel
:2dense_58/bias
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
(
è0"
trackable_list_wrapper
²
non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_58_layer_call_fn_1770008¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
E__inference_dense_58_layer_call_and_return_conditional_losses_1770024¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
*:(2batch_normalization_52/gamma
):'2batch_normalization_52/beta
2:0 (2"batch_normalization_52/moving_mean
6:4 (2&batch_normalization_52/moving_variance
>
~0
1
2
3"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_52_layer_call_fn_1770037
8__inference_batch_normalization_52_layer_call_fn_1770050´
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
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1770070
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1770104´
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
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_52_layer_call_fn_1770109¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
K__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_1770114¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:2dense_59/kernel
:2dense_59/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
(
é0"
trackable_list_wrapper
¸
®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_59_layer_call_fn_1770129¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
E__inference_dense_59_layer_call_and_return_conditional_losses_1770145¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
*:(2batch_normalization_53/gamma
):'2batch_normalization_53/beta
2:0 (2"batch_normalization_53/moving_mean
6:4 (2&batch_normalization_53/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_53_layer_call_fn_1770158
8__inference_batch_normalization_53_layer_call_fn_1770171´
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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1770191
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1770225´
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
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_53_layer_call_fn_1770230¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
K__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_1770235¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:2dense_60/kernel
:2dense_60/bias
0
§0
¨1"
trackable_list_wrapper
0
§0
¨1"
trackable_list_wrapper
(
ê0"
trackable_list_wrapper
¸
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
©	variables
ªtrainable_variables
«regularization_losses
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_60_layer_call_fn_1770250¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
E__inference_dense_60_layer_call_and_return_conditional_losses_1770266¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
*:(2batch_normalization_54/gamma
):'2batch_normalization_54/beta
2:0 (2"batch_normalization_54/moving_mean
6:4 (2&batch_normalization_54/moving_variance
@
°0
±1
²2
³3"
trackable_list_wrapper
0
°0
±1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_54_layer_call_fn_1770279
8__inference_batch_normalization_54_layer_call_fn_1770292´
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
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1770312
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1770346´
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
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
º	variables
»trainable_variables
¼regularization_losses
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_54_layer_call_fn_1770351¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
K__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_1770356¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:2dense_61/kernel
:2dense_61/bias
0
À0
Á1"
trackable_list_wrapper
0
À0
Á1"
trackable_list_wrapper
(
ë0"
trackable_list_wrapper
¸
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
Â	variables
Ãtrainable_variables
Äregularization_losses
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_61_layer_call_fn_1770371¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
E__inference_dense_61_layer_call_and_return_conditional_losses_1770387¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
*:(2batch_normalization_55/gamma
):'2batch_normalization_55/beta
2:0 (2"batch_normalization_55/moving_mean
6:4 (2&batch_normalization_55/moving_variance
@
É0
Ê1
Ë2
Ì3"
trackable_list_wrapper
0
É0
Ê1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
Í	variables
Îtrainable_variables
Ïregularization_losses
Ñ__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
®2«
8__inference_batch_normalization_55_layer_call_fn_1770400
8__inference_batch_normalization_55_layer_call_fn_1770413´
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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1770433
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1770467´
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
Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
Ó	variables
Ôtrainable_variables
Õregularization_losses
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_leaky_re_lu_55_layer_call_fn_1770472¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
K__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_1770477¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:2dense_62/kernel
:2dense_62/bias
0
Ù0
Ú1"
trackable_list_wrapper
0
Ù0
Ú1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_dense_62_layer_call_fn_1770486¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
E__inference_dense_62_layer_call_and_return_conditional_losses_1770496¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
__inference_loss_fn_0_1770507
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
__inference_loss_fn_1_1770518
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
__inference_loss_fn_2_1770529
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
__inference_loss_fn_3_1770540
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
__inference_loss_fn_4_1770551
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
__inference_loss_fn_5_1770562
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
__inference_loss_fn_6_1770573
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
¦
%0
&1
'2
53
64
N5
O6
g7
h8
9
10
11
12
²13
³14
Ë15
Ì16"
trackable_list_wrapper
Î
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
22"
trackable_list_wrapper
(
à0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÚB×
%__inference_signature_wrapper_1769583normalization_6_input"
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
å0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
50
61"
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
æ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
N0
O1"
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
ç0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
g0
h1"
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
è0"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
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
é0"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
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
ê0"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
²0
³1"
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
ë0"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ë0
Ì1"
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

átotal

âcount
ã	variables
ä	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
á0
â1"
trackable_list_wrapper
.
ã	variables"
_generic_user_object
&:$72Adam/dense_55/kernel/m
 :72Adam/dense_55/bias/m
/:-72#Adam/batch_normalization_49/gamma/m
.:,72"Adam/batch_normalization_49/beta/m
&:$772Adam/dense_56/kernel/m
 :72Adam/dense_56/bias/m
/:-72#Adam/batch_normalization_50/gamma/m
.:,72"Adam/batch_normalization_50/beta/m
&:$7u2Adam/dense_57/kernel/m
 :u2Adam/dense_57/bias/m
/:-u2#Adam/batch_normalization_51/gamma/m
.:,u2"Adam/batch_normalization_51/beta/m
&:$u2Adam/dense_58/kernel/m
 :2Adam/dense_58/bias/m
/:-2#Adam/batch_normalization_52/gamma/m
.:,2"Adam/batch_normalization_52/beta/m
&:$2Adam/dense_59/kernel/m
 :2Adam/dense_59/bias/m
/:-2#Adam/batch_normalization_53/gamma/m
.:,2"Adam/batch_normalization_53/beta/m
&:$2Adam/dense_60/kernel/m
 :2Adam/dense_60/bias/m
/:-2#Adam/batch_normalization_54/gamma/m
.:,2"Adam/batch_normalization_54/beta/m
&:$2Adam/dense_61/kernel/m
 :2Adam/dense_61/bias/m
/:-2#Adam/batch_normalization_55/gamma/m
.:,2"Adam/batch_normalization_55/beta/m
&:$2Adam/dense_62/kernel/m
 :2Adam/dense_62/bias/m
&:$72Adam/dense_55/kernel/v
 :72Adam/dense_55/bias/v
/:-72#Adam/batch_normalization_49/gamma/v
.:,72"Adam/batch_normalization_49/beta/v
&:$772Adam/dense_56/kernel/v
 :72Adam/dense_56/bias/v
/:-72#Adam/batch_normalization_50/gamma/v
.:,72"Adam/batch_normalization_50/beta/v
&:$7u2Adam/dense_57/kernel/v
 :u2Adam/dense_57/bias/v
/:-u2#Adam/batch_normalization_51/gamma/v
.:,u2"Adam/batch_normalization_51/beta/v
&:$u2Adam/dense_58/kernel/v
 :2Adam/dense_58/bias/v
/:-2#Adam/batch_normalization_52/gamma/v
.:,2"Adam/batch_normalization_52/beta/v
&:$2Adam/dense_59/kernel/v
 :2Adam/dense_59/bias/v
/:-2#Adam/batch_normalization_53/gamma/v
.:,2"Adam/batch_normalization_53/beta/v
&:$2Adam/dense_60/kernel/v
 :2Adam/dense_60/bias/v
/:-2#Adam/batch_normalization_54/gamma/v
.:,2"Adam/batch_normalization_54/beta/v
&:$2Adam/dense_61/kernel/v
 :2Adam/dense_61/bias/v
/:-2#Adam/batch_normalization_55/gamma/v
.:,2"Adam/batch_normalization_55/beta/v
&:$2Adam/dense_62/kernel/v
 :2Adam/dense_62/bias/v
	J
Const
J	
Const_1ä
"__inference__wrapped_model_1766796½F¡¢*+6354CDOLNM\]hegfuv~§¨³°²±ÀÁÌÉËÊÙÚ>¢;
4¢1
/,
normalization_6_inputÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_62"
dense_62ÿÿÿÿÿÿÿÿÿp
__inference_adapt_step_1769630N'%&C¢@
9¢6
41¢
ÿÿÿÿÿÿÿÿÿIteratorSpec 
ª "
 ¹
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1769707b63543¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 ¹
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1769741b56343¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 
8__inference_batch_normalization_49_layer_call_fn_1769674U63543¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p 
ª "ÿÿÿÿÿÿÿÿÿ7
8__inference_batch_normalization_49_layer_call_fn_1769687U56343¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p
ª "ÿÿÿÿÿÿÿÿÿ7¹
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1769828bOLNM3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 ¹
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1769862bNOLM3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 
8__inference_batch_normalization_50_layer_call_fn_1769795UOLNM3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p 
ª "ÿÿÿÿÿÿÿÿÿ7
8__inference_batch_normalization_50_layer_call_fn_1769808UNOLM3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p
ª "ÿÿÿÿÿÿÿÿÿ7¹
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1769949bhegf3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿu
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿu
 ¹
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1769983bghef3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿu
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿu
 
8__inference_batch_normalization_51_layer_call_fn_1769916Uhegf3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿu
p 
ª "ÿÿÿÿÿÿÿÿÿu
8__inference_batch_normalization_51_layer_call_fn_1769929Ughef3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿu
p
ª "ÿÿÿÿÿÿÿÿÿu»
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1770070d~3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 »
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1770104d~3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_52_layer_call_fn_1770037W~3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_52_layer_call_fn_1770050W~3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ½
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1770191f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1770225f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_53_layer_call_fn_1770158Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_53_layer_call_fn_1770171Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ½
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1770312f³°²±3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1770346f²³°±3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_54_layer_call_fn_1770279Y³°²±3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_54_layer_call_fn_1770292Y²³°±3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ½
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1770433fÌÉËÊ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1770467fËÌÉÊ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_55_layer_call_fn_1770400YÌÉËÊ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_55_layer_call_fn_1770413YËÌÉÊ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_55_layer_call_and_return_conditional_losses_1769661\*+/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 }
*__inference_dense_55_layer_call_fn_1769645O*+/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ7¥
E__inference_dense_56_layer_call_and_return_conditional_losses_1769782\CD/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 }
*__inference_dense_56_layer_call_fn_1769766OCD/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "ÿÿÿÿÿÿÿÿÿ7¥
E__inference_dense_57_layer_call_and_return_conditional_losses_1769903\\]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "%¢"

0ÿÿÿÿÿÿÿÿÿu
 }
*__inference_dense_57_layer_call_fn_1769887O\]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "ÿÿÿÿÿÿÿÿÿu¥
E__inference_dense_58_layer_call_and_return_conditional_losses_1770024\uv/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿu
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_58_layer_call_fn_1770008Ouv/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿu
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_59_layer_call_and_return_conditional_losses_1770145^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_59_layer_call_fn_1770129Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_60_layer_call_and_return_conditional_losses_1770266^§¨/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_60_layer_call_fn_1770250Q§¨/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_61_layer_call_and_return_conditional_losses_1770387^ÀÁ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_61_layer_call_fn_1770371QÀÁ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_62_layer_call_and_return_conditional_losses_1770496^ÙÚ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_62_layer_call_fn_1770486QÙÚ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_1769751X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 
0__inference_leaky_re_lu_49_layer_call_fn_1769746K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "ÿÿÿÿÿÿÿÿÿ7§
K__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_1769872X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 
0__inference_leaky_re_lu_50_layer_call_fn_1769867K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "ÿÿÿÿÿÿÿÿÿ7§
K__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_1769993X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿu
ª "%¢"

0ÿÿÿÿÿÿÿÿÿu
 
0__inference_leaky_re_lu_51_layer_call_fn_1769988K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿu
ª "ÿÿÿÿÿÿÿÿÿu§
K__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_1770114X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_leaky_re_lu_52_layer_call_fn_1770109K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_1770235X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_leaky_re_lu_53_layer_call_fn_1770230K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_1770356X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_leaky_re_lu_54_layer_call_fn_1770351K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
K__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_1770477X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_leaky_re_lu_55_layer_call_fn_1770472K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ<
__inference_loss_fn_0_1770507*¢

¢ 
ª " <
__inference_loss_fn_1_1770518C¢

¢ 
ª " <
__inference_loss_fn_2_1770529\¢

¢ 
ª " <
__inference_loss_fn_3_1770540u¢

¢ 
ª " =
__inference_loss_fn_4_1770551¢

¢ 
ª " =
__inference_loss_fn_5_1770562§¢

¢ 
ª " =
__inference_loss_fn_6_1770573À¢

¢ 
ª " 
I__inference_sequential_6_layer_call_and_return_conditional_losses_1768543·F¡¢*+6354CDOLNM\]hegfuv~§¨³°²±ÀÁÌÉËÊÙÚF¢C
<¢9
/,
normalization_6_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
I__inference_sequential_6_layer_call_and_return_conditional_losses_1768706·F¡¢*+5634CDNOLM\]ghefuv~§¨²³°±ÀÁËÌÉÊÙÚF¢C
<¢9
/,
normalization_6_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ö
I__inference_sequential_6_layer_call_and_return_conditional_losses_1769166¨F¡¢*+6354CDOLNM\]hegfuv~§¨³°²±ÀÁÌÉËÊÙÚ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ö
I__inference_sequential_6_layer_call_and_return_conditional_losses_1769484¨F¡¢*+5634CDNOLM\]ghefuv~§¨²³°±ÀÁËÌÉÊÙÚ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ý
.__inference_sequential_6_layer_call_fn_1767804ªF¡¢*+6354CDOLNM\]hegfuv~§¨³°²±ÀÁÌÉËÊÙÚF¢C
<¢9
/,
normalization_6_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÝ
.__inference_sequential_6_layer_call_fn_1768380ªF¡¢*+5634CDNOLM\]ghefuv~§¨²³°±ÀÁËÌÉÊÙÚF¢C
<¢9
/,
normalization_6_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÎ
.__inference_sequential_6_layer_call_fn_1768849F¡¢*+6354CDOLNM\]hegfuv~§¨³°²±ÀÁÌÉËÊÙÚ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÎ
.__inference_sequential_6_layer_call_fn_1768946F¡¢*+5634CDNOLM\]ghefuv~§¨²³°±ÀÁËÌÉÊÙÚ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_signature_wrapper_1769583ÖF¡¢*+6354CDOLNM\]hegfuv~§¨³°²±ÀÁÌÉËÊÙÚW¢T
¢ 
MªJ
H
normalization_6_input/,
normalization_6_inputÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_62"
dense_62ÿÿÿÿÿÿÿÿÿ