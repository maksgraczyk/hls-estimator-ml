¬1
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Ô,
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
~
dense_1001/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:5*"
shared_namedense_1001/kernel
w
%dense_1001/kernel/Read/ReadVariableOpReadVariableOpdense_1001/kernel*
_output_shapes

:5*
dtype0
v
dense_1001/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:5* 
shared_namedense_1001/bias
o
#dense_1001/bias/Read/ReadVariableOpReadVariableOpdense_1001/bias*
_output_shapes
:5*
dtype0

batch_normalization_904/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*.
shared_namebatch_normalization_904/gamma

1batch_normalization_904/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_904/gamma*
_output_shapes
:5*
dtype0

batch_normalization_904/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*-
shared_namebatch_normalization_904/beta

0batch_normalization_904/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_904/beta*
_output_shapes
:5*
dtype0

#batch_normalization_904/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#batch_normalization_904/moving_mean

7batch_normalization_904/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_904/moving_mean*
_output_shapes
:5*
dtype0
¦
'batch_normalization_904/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*8
shared_name)'batch_normalization_904/moving_variance

;batch_normalization_904/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_904/moving_variance*
_output_shapes
:5*
dtype0
~
dense_1002/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:55*"
shared_namedense_1002/kernel
w
%dense_1002/kernel/Read/ReadVariableOpReadVariableOpdense_1002/kernel*
_output_shapes

:55*
dtype0
v
dense_1002/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:5* 
shared_namedense_1002/bias
o
#dense_1002/bias/Read/ReadVariableOpReadVariableOpdense_1002/bias*
_output_shapes
:5*
dtype0

batch_normalization_905/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*.
shared_namebatch_normalization_905/gamma

1batch_normalization_905/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_905/gamma*
_output_shapes
:5*
dtype0

batch_normalization_905/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*-
shared_namebatch_normalization_905/beta

0batch_normalization_905/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_905/beta*
_output_shapes
:5*
dtype0

#batch_normalization_905/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#batch_normalization_905/moving_mean

7batch_normalization_905/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_905/moving_mean*
_output_shapes
:5*
dtype0
¦
'batch_normalization_905/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*8
shared_name)'batch_normalization_905/moving_variance

;batch_normalization_905/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_905/moving_variance*
_output_shapes
:5*
dtype0
~
dense_1003/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:5*"
shared_namedense_1003/kernel
w
%dense_1003/kernel/Read/ReadVariableOpReadVariableOpdense_1003/kernel*
_output_shapes

:5*
dtype0
v
dense_1003/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_1003/bias
o
#dense_1003/bias/Read/ReadVariableOpReadVariableOpdense_1003/bias*
_output_shapes
:*
dtype0

batch_normalization_906/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_906/gamma

1batch_normalization_906/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_906/gamma*
_output_shapes
:*
dtype0

batch_normalization_906/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_906/beta

0batch_normalization_906/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_906/beta*
_output_shapes
:*
dtype0

#batch_normalization_906/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_906/moving_mean

7batch_normalization_906/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_906/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_906/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_906/moving_variance

;batch_normalization_906/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_906/moving_variance*
_output_shapes
:*
dtype0
~
dense_1004/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_1004/kernel
w
%dense_1004/kernel/Read/ReadVariableOpReadVariableOpdense_1004/kernel*
_output_shapes

:*
dtype0
v
dense_1004/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_1004/bias
o
#dense_1004/bias/Read/ReadVariableOpReadVariableOpdense_1004/bias*
_output_shapes
:*
dtype0

batch_normalization_907/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_907/gamma

1batch_normalization_907/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_907/gamma*
_output_shapes
:*
dtype0

batch_normalization_907/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_907/beta

0batch_normalization_907/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_907/beta*
_output_shapes
:*
dtype0

#batch_normalization_907/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_907/moving_mean

7batch_normalization_907/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_907/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_907/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_907/moving_variance

;batch_normalization_907/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_907/moving_variance*
_output_shapes
:*
dtype0
~
dense_1005/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_1005/kernel
w
%dense_1005/kernel/Read/ReadVariableOpReadVariableOpdense_1005/kernel*
_output_shapes

:*
dtype0
v
dense_1005/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_1005/bias
o
#dense_1005/bias/Read/ReadVariableOpReadVariableOpdense_1005/bias*
_output_shapes
:*
dtype0

batch_normalization_908/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_908/gamma

1batch_normalization_908/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_908/gamma*
_output_shapes
:*
dtype0

batch_normalization_908/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_908/beta

0batch_normalization_908/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_908/beta*
_output_shapes
:*
dtype0

#batch_normalization_908/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_908/moving_mean

7batch_normalization_908/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_908/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_908/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_908/moving_variance

;batch_normalization_908/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_908/moving_variance*
_output_shapes
:*
dtype0
~
dense_1006/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_1006/kernel
w
%dense_1006/kernel/Read/ReadVariableOpReadVariableOpdense_1006/kernel*
_output_shapes

:*
dtype0
v
dense_1006/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_1006/bias
o
#dense_1006/bias/Read/ReadVariableOpReadVariableOpdense_1006/bias*
_output_shapes
:*
dtype0

batch_normalization_909/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_909/gamma

1batch_normalization_909/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_909/gamma*
_output_shapes
:*
dtype0

batch_normalization_909/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_909/beta

0batch_normalization_909/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_909/beta*
_output_shapes
:*
dtype0

#batch_normalization_909/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_909/moving_mean

7batch_normalization_909/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_909/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_909/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_909/moving_variance

;batch_normalization_909/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_909/moving_variance*
_output_shapes
:*
dtype0
~
dense_1007/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_1007/kernel
w
%dense_1007/kernel/Read/ReadVariableOpReadVariableOpdense_1007/kernel*
_output_shapes

:*
dtype0
v
dense_1007/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_1007/bias
o
#dense_1007/bias/Read/ReadVariableOpReadVariableOpdense_1007/bias*
_output_shapes
:*
dtype0

batch_normalization_910/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_910/gamma

1batch_normalization_910/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_910/gamma*
_output_shapes
:*
dtype0

batch_normalization_910/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_910/beta

0batch_normalization_910/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_910/beta*
_output_shapes
:*
dtype0

#batch_normalization_910/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_910/moving_mean

7batch_normalization_910/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_910/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_910/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_910/moving_variance

;batch_normalization_910/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_910/moving_variance*
_output_shapes
:*
dtype0
~
dense_1008/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_1008/kernel
w
%dense_1008/kernel/Read/ReadVariableOpReadVariableOpdense_1008/kernel*
_output_shapes

:*
dtype0
v
dense_1008/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_1008/bias
o
#dense_1008/bias/Read/ReadVariableOpReadVariableOpdense_1008/bias*
_output_shapes
:*
dtype0

batch_normalization_911/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_911/gamma

1batch_normalization_911/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_911/gamma*
_output_shapes
:*
dtype0

batch_normalization_911/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_911/beta

0batch_normalization_911/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_911/beta*
_output_shapes
:*
dtype0

#batch_normalization_911/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_911/moving_mean

7batch_normalization_911/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_911/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_911/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_911/moving_variance

;batch_normalization_911/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_911/moving_variance*
_output_shapes
:*
dtype0
~
dense_1009/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_1009/kernel
w
%dense_1009/kernel/Read/ReadVariableOpReadVariableOpdense_1009/kernel*
_output_shapes

:*
dtype0
v
dense_1009/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_1009/bias
o
#dense_1009/bias/Read/ReadVariableOpReadVariableOpdense_1009/bias*
_output_shapes
:*
dtype0

batch_normalization_912/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_912/gamma

1batch_normalization_912/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_912/gamma*
_output_shapes
:*
dtype0

batch_normalization_912/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_912/beta

0batch_normalization_912/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_912/beta*
_output_shapes
:*
dtype0

#batch_normalization_912/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_912/moving_mean

7batch_normalization_912/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_912/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_912/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_912/moving_variance

;batch_normalization_912/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_912/moving_variance*
_output_shapes
:*
dtype0
~
dense_1010/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_1010/kernel
w
%dense_1010/kernel/Read/ReadVariableOpReadVariableOpdense_1010/kernel*
_output_shapes

:*
dtype0
v
dense_1010/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_1010/bias
o
#dense_1010/bias/Read/ReadVariableOpReadVariableOpdense_1010/bias*
_output_shapes
:*
dtype0

batch_normalization_913/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_913/gamma

1batch_normalization_913/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_913/gamma*
_output_shapes
:*
dtype0

batch_normalization_913/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_913/beta

0batch_normalization_913/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_913/beta*
_output_shapes
:*
dtype0

#batch_normalization_913/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_913/moving_mean

7batch_normalization_913/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_913/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_913/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_913/moving_variance

;batch_normalization_913/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_913/moving_variance*
_output_shapes
:*
dtype0
~
dense_1011/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_1011/kernel
w
%dense_1011/kernel/Read/ReadVariableOpReadVariableOpdense_1011/kernel*
_output_shapes

:*
dtype0
v
dense_1011/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_1011/bias
o
#dense_1011/bias/Read/ReadVariableOpReadVariableOpdense_1011/bias*
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

Adam/dense_1001/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:5*)
shared_nameAdam/dense_1001/kernel/m

,Adam/dense_1001/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1001/kernel/m*
_output_shapes

:5*
dtype0

Adam/dense_1001/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*'
shared_nameAdam/dense_1001/bias/m
}
*Adam/dense_1001/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1001/bias/m*
_output_shapes
:5*
dtype0
 
$Adam/batch_normalization_904/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*5
shared_name&$Adam/batch_normalization_904/gamma/m

8Adam/batch_normalization_904/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_904/gamma/m*
_output_shapes
:5*
dtype0

#Adam/batch_normalization_904/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#Adam/batch_normalization_904/beta/m

7Adam/batch_normalization_904/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_904/beta/m*
_output_shapes
:5*
dtype0

Adam/dense_1002/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:55*)
shared_nameAdam/dense_1002/kernel/m

,Adam/dense_1002/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1002/kernel/m*
_output_shapes

:55*
dtype0

Adam/dense_1002/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*'
shared_nameAdam/dense_1002/bias/m
}
*Adam/dense_1002/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1002/bias/m*
_output_shapes
:5*
dtype0
 
$Adam/batch_normalization_905/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*5
shared_name&$Adam/batch_normalization_905/gamma/m

8Adam/batch_normalization_905/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_905/gamma/m*
_output_shapes
:5*
dtype0

#Adam/batch_normalization_905/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#Adam/batch_normalization_905/beta/m

7Adam/batch_normalization_905/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_905/beta/m*
_output_shapes
:5*
dtype0

Adam/dense_1003/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:5*)
shared_nameAdam/dense_1003/kernel/m

,Adam/dense_1003/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1003/kernel/m*
_output_shapes

:5*
dtype0

Adam/dense_1003/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1003/bias/m
}
*Adam/dense_1003/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1003/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_906/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_906/gamma/m

8Adam/batch_normalization_906/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_906/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_906/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_906/beta/m

7Adam/batch_normalization_906/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_906/beta/m*
_output_shapes
:*
dtype0

Adam/dense_1004/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1004/kernel/m

,Adam/dense_1004/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1004/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_1004/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1004/bias/m
}
*Adam/dense_1004/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1004/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_907/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_907/gamma/m

8Adam/batch_normalization_907/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_907/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_907/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_907/beta/m

7Adam/batch_normalization_907/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_907/beta/m*
_output_shapes
:*
dtype0

Adam/dense_1005/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1005/kernel/m

,Adam/dense_1005/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1005/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_1005/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1005/bias/m
}
*Adam/dense_1005/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1005/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_908/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_908/gamma/m

8Adam/batch_normalization_908/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_908/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_908/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_908/beta/m

7Adam/batch_normalization_908/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_908/beta/m*
_output_shapes
:*
dtype0

Adam/dense_1006/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1006/kernel/m

,Adam/dense_1006/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1006/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_1006/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1006/bias/m
}
*Adam/dense_1006/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1006/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_909/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_909/gamma/m

8Adam/batch_normalization_909/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_909/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_909/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_909/beta/m

7Adam/batch_normalization_909/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_909/beta/m*
_output_shapes
:*
dtype0

Adam/dense_1007/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1007/kernel/m

,Adam/dense_1007/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1007/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_1007/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1007/bias/m
}
*Adam/dense_1007/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1007/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_910/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_910/gamma/m

8Adam/batch_normalization_910/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_910/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_910/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_910/beta/m

7Adam/batch_normalization_910/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_910/beta/m*
_output_shapes
:*
dtype0

Adam/dense_1008/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1008/kernel/m

,Adam/dense_1008/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1008/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_1008/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1008/bias/m
}
*Adam/dense_1008/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1008/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_911/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_911/gamma/m

8Adam/batch_normalization_911/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_911/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_911/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_911/beta/m

7Adam/batch_normalization_911/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_911/beta/m*
_output_shapes
:*
dtype0

Adam/dense_1009/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1009/kernel/m

,Adam/dense_1009/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1009/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_1009/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1009/bias/m
}
*Adam/dense_1009/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1009/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_912/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_912/gamma/m

8Adam/batch_normalization_912/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_912/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_912/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_912/beta/m

7Adam/batch_normalization_912/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_912/beta/m*
_output_shapes
:*
dtype0

Adam/dense_1010/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1010/kernel/m

,Adam/dense_1010/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1010/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_1010/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1010/bias/m
}
*Adam/dense_1010/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1010/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_913/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_913/gamma/m

8Adam/batch_normalization_913/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_913/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_913/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_913/beta/m

7Adam/batch_normalization_913/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_913/beta/m*
_output_shapes
:*
dtype0

Adam/dense_1011/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1011/kernel/m

,Adam/dense_1011/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1011/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_1011/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1011/bias/m
}
*Adam/dense_1011/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1011/bias/m*
_output_shapes
:*
dtype0

Adam/dense_1001/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:5*)
shared_nameAdam/dense_1001/kernel/v

,Adam/dense_1001/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1001/kernel/v*
_output_shapes

:5*
dtype0

Adam/dense_1001/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*'
shared_nameAdam/dense_1001/bias/v
}
*Adam/dense_1001/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1001/bias/v*
_output_shapes
:5*
dtype0
 
$Adam/batch_normalization_904/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*5
shared_name&$Adam/batch_normalization_904/gamma/v

8Adam/batch_normalization_904/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_904/gamma/v*
_output_shapes
:5*
dtype0

#Adam/batch_normalization_904/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#Adam/batch_normalization_904/beta/v

7Adam/batch_normalization_904/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_904/beta/v*
_output_shapes
:5*
dtype0

Adam/dense_1002/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:55*)
shared_nameAdam/dense_1002/kernel/v

,Adam/dense_1002/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1002/kernel/v*
_output_shapes

:55*
dtype0

Adam/dense_1002/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*'
shared_nameAdam/dense_1002/bias/v
}
*Adam/dense_1002/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1002/bias/v*
_output_shapes
:5*
dtype0
 
$Adam/batch_normalization_905/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*5
shared_name&$Adam/batch_normalization_905/gamma/v

8Adam/batch_normalization_905/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_905/gamma/v*
_output_shapes
:5*
dtype0

#Adam/batch_normalization_905/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*4
shared_name%#Adam/batch_normalization_905/beta/v

7Adam/batch_normalization_905/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_905/beta/v*
_output_shapes
:5*
dtype0

Adam/dense_1003/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:5*)
shared_nameAdam/dense_1003/kernel/v

,Adam/dense_1003/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1003/kernel/v*
_output_shapes

:5*
dtype0

Adam/dense_1003/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1003/bias/v
}
*Adam/dense_1003/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1003/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_906/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_906/gamma/v

8Adam/batch_normalization_906/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_906/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_906/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_906/beta/v

7Adam/batch_normalization_906/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_906/beta/v*
_output_shapes
:*
dtype0

Adam/dense_1004/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1004/kernel/v

,Adam/dense_1004/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1004/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_1004/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1004/bias/v
}
*Adam/dense_1004/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1004/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_907/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_907/gamma/v

8Adam/batch_normalization_907/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_907/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_907/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_907/beta/v

7Adam/batch_normalization_907/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_907/beta/v*
_output_shapes
:*
dtype0

Adam/dense_1005/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1005/kernel/v

,Adam/dense_1005/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1005/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_1005/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1005/bias/v
}
*Adam/dense_1005/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1005/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_908/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_908/gamma/v

8Adam/batch_normalization_908/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_908/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_908/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_908/beta/v

7Adam/batch_normalization_908/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_908/beta/v*
_output_shapes
:*
dtype0

Adam/dense_1006/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1006/kernel/v

,Adam/dense_1006/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1006/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_1006/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1006/bias/v
}
*Adam/dense_1006/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1006/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_909/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_909/gamma/v

8Adam/batch_normalization_909/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_909/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_909/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_909/beta/v

7Adam/batch_normalization_909/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_909/beta/v*
_output_shapes
:*
dtype0

Adam/dense_1007/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1007/kernel/v

,Adam/dense_1007/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1007/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_1007/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1007/bias/v
}
*Adam/dense_1007/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1007/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_910/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_910/gamma/v

8Adam/batch_normalization_910/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_910/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_910/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_910/beta/v

7Adam/batch_normalization_910/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_910/beta/v*
_output_shapes
:*
dtype0

Adam/dense_1008/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1008/kernel/v

,Adam/dense_1008/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1008/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_1008/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1008/bias/v
}
*Adam/dense_1008/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1008/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_911/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_911/gamma/v

8Adam/batch_normalization_911/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_911/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_911/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_911/beta/v

7Adam/batch_normalization_911/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_911/beta/v*
_output_shapes
:*
dtype0

Adam/dense_1009/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1009/kernel/v

,Adam/dense_1009/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1009/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_1009/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1009/bias/v
}
*Adam/dense_1009/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1009/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_912/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_912/gamma/v

8Adam/batch_normalization_912/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_912/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_912/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_912/beta/v

7Adam/batch_normalization_912/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_912/beta/v*
_output_shapes
:*
dtype0

Adam/dense_1010/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1010/kernel/v

,Adam/dense_1010/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1010/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_1010/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1010/bias/v
}
*Adam/dense_1010/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1010/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_913/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_913/gamma/v

8Adam/batch_normalization_913/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_913/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_913/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_913/beta/v

7Adam/batch_normalization_913/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_913/beta/v*
_output_shapes
:*
dtype0

Adam/dense_1011/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/dense_1011/kernel/v

,Adam/dense_1011/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1011/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_1011/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1011/bias/v
}
*Adam/dense_1011/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1011/bias/v*
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
½
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*Ê¼
value¿¼B»¼ B³¼
Ê	
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
!	optimizer
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(_default_save_signature
)
signatures*
¾
*
_keep_axis
+_reduce_axis
,_reduce_axis_mask
-_broadcast_shape
.mean
.
adapt_mean
/variance
/adapt_variance
	0count
1	keras_api
2_adapt_function*
¦

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses*
Õ
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses*

F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses* 
¦

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses*
Õ
Taxis
	Ugamma
Vbeta
Wmoving_mean
Xmoving_variance
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses*

_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses* 
¦

ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses*
Õ
maxis
	ngamma
obeta
pmoving_mean
qmoving_variance
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses*

x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses* 
¬

~kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

 gamma
	¡beta
¢moving_mean
£moving_variance
¤	variables
¥trainable_variables
¦regularization_losses
§	keras_api
¨__call__
+©&call_and_return_all_conditional_losses*

ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses* 
®
°kernel
	±bias
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses*
à
	¸axis

¹gamma
	ºbeta
»moving_mean
¼moving_variance
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses*

Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses* 
®
Ékernel
	Êbias
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses*
à
	Ñaxis

Ògamma
	Óbeta
Ômoving_mean
Õmoving_variance
Ö	variables
×trainable_variables
Øregularization_losses
Ù	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses*

Ü	variables
Ýtrainable_variables
Þregularization_losses
ß	keras_api
à__call__
+á&call_and_return_all_conditional_losses* 
®
âkernel
	ãbias
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
è__call__
+é&call_and_return_all_conditional_losses*
à
	êaxis

ëgamma
	ìbeta
ímoving_mean
îmoving_variance
ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses*

õ	variables
ötrainable_variables
÷regularization_losses
ø	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses* 
®
ûkernel
	übias
ý	variables
þtrainable_variables
ÿregularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
à
	axis

gamma
	beta
moving_mean
 moving_variance
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses*

§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses* 
®
­kernel
	®bias
¯	variables
°trainable_variables
±regularization_losses
²	keras_api
³__call__
+´&call_and_return_all_conditional_losses*
µ
	µiter
¶beta_1
·beta_2

¸decay3mß4mà<má=mâLmãMmäUmåVmæemçfmènméomê~mëmì	mí	mî	mï	mð	 mñ	¡mò	°mó	±mô	¹mõ	ºmö	Ém÷	Êmø	Òmù	Ómú	âmû	ãmü	ëmý	ìmþ	ûmÿ	üm	m	m	m	m	m	m	­m	®m3v4v<v=vLvMvUvVvevfvnvov~vv	v	v	v	v	 v	¡v	°v	±v	¹v	ºv 	Év¡	Êv¢	Òv£	Óv¤	âv¥	ãv¦	ëv§	ìv¨	ûv©	üvª	v«	v¬	v­	v®	v¯	v°	­v±	®v²*
¬
.0
/1
02
33
44
<5
=6
>7
?8
L9
M10
U11
V12
W13
X14
e15
f16
n17
o18
p19
q20
~21
22
23
24
25
26
27
28
 29
¡30
¢31
£32
°33
±34
¹35
º36
»37
¼38
É39
Ê40
Ò41
Ó42
Ô43
Õ44
â45
ã46
ë47
ì48
í49
î50
û51
ü52
53
54
55
56
57
58
59
60
61
 62
­63
®64*
æ
30
41
<2
=3
L4
M5
U6
V7
e8
f9
n10
o11
~12
13
14
15
16
17
 18
¡19
°20
±21
¹22
º23
É24
Ê25
Ò26
Ó27
â28
ã29
ë30
ì31
û32
ü33
34
35
36
37
38
39
­40
®41*
* 
µ
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
(_default_save_signature
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 
* 

¾serving_default* 
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
a[
VARIABLE_VALUEdense_1001/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_1001/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

30
41*

30
41*
* 

¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_904/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_904/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_904/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_904/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
<0
=1
>2
?3*

<0
=1*
* 

Änon_trainable_variables
Ålayers
Æmetrics
 Çlayer_regularization_losses
Èlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_1002/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_1002/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

L0
M1*

L0
M1*
* 

Înon_trainable_variables
Ïlayers
Ðmetrics
 Ñlayer_regularization_losses
Òlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_905/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_905/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_905/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_905/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
U0
V1
W2
X3*

U0
V1*
* 

Ónon_trainable_variables
Ôlayers
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ønon_trainable_variables
Ùlayers
Úmetrics
 Ûlayer_regularization_losses
Ülayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_1003/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_1003/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

e0
f1*

e0
f1*
* 

Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_906/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_906/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_906/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_906/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
n0
o1
p2
q3*

n0
o1*
* 

ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_1004/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_1004/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

~0
1*

~0
1*
* 

ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_907/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_907/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_907/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_907/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

ñnon_trainable_variables
òlayers
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

önon_trainable_variables
÷layers
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
a[
VARIABLE_VALUEdense_1005/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_1005/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_908/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_908/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_908/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_908/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
 0
¡1
¢2
£3*

 0
¡1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¤	variables
¥trainable_variables
¦regularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses* 
* 
* 
b\
VARIABLE_VALUEdense_1006/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_1006/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

°0
±1*

°0
±1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_909/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_909/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_909/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_909/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
¹0
º1
»2
¼3*

¹0
º1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses* 
* 
* 
b\
VARIABLE_VALUEdense_1007/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_1007/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

É0
Ê1*

É0
Ê1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_910/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_910/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_910/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_910/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
Ò0
Ó1
Ô2
Õ3*

Ò0
Ó1*
* 

non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
Ö	variables
×trainable_variables
Øregularization_losses
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
Ü	variables
Ýtrainable_variables
Þregularization_losses
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses* 
* 
* 
b\
VARIABLE_VALUEdense_1008/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_1008/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

â0
ã1*

â0
ã1*
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
ä	variables
åtrainable_variables
æregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_911/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_911/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_911/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_911/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
ë0
ì1
í2
î3*

ë0
ì1*
* 

­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
ï	variables
ðtrainable_variables
ñregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
õ	variables
ötrainable_variables
÷regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses* 
* 
* 
b\
VARIABLE_VALUEdense_1009/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_1009/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*

û0
ü1*

û0
ü1*
* 

·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
ý	variables
þtrainable_variables
ÿregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_912/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_912/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_912/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_912/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
b\
VARIABLE_VALUEdense_1010/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_1010/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEbatch_normalization_913/gamma6layer_with_weights-20/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEbatch_normalization_913/beta5layer_with_weights-20/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#batch_normalization_913/moving_mean<layer_with_weights-20/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE'batch_normalization_913/moving_variance@layer_with_weights-20/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
 3*

0
1*
* 

Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses* 
* 
* 
b\
VARIABLE_VALUEdense_1011/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdense_1011/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE*

­0
®1*

­0
®1*
* 

Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
¯	variables
°trainable_variables
±regularization_losses
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses*
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
À
.0
/1
02
>3
?4
W5
X6
p7
q8
9
10
¢11
£12
»13
¼14
Ô15
Õ16
í17
î18
19
20
21
 22*
ú
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
 31*

Ú0*
* 
* 
* 
* 
* 
* 
* 
* 

>0
?1*
* 
* 
* 
* 
* 
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
W0
X1*
* 
* 
* 
* 
* 
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
p0
q1*
* 
* 
* 
* 
* 
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
0
1*
* 
* 
* 
* 
* 
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
¢0
£1*
* 
* 
* 
* 
* 
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
»0
¼1*
* 
* 
* 
* 
* 
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
Ô0
Õ1*
* 
* 
* 
* 
* 
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
í0
î1*
* 
* 
* 
* 
* 
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
0
1*
* 
* 
* 
* 
* 
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
0
 1*
* 
* 
* 
* 
* 
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

Ûtotal

Ücount
Ý	variables
Þ	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Û0
Ü1*

Ý	variables*
~
VARIABLE_VALUEAdam/dense_1001/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1001/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_904/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_904/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1002/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1002/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_905/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_905/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1003/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1003/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_906/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_906/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1004/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1004/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_907/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_907/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1005/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1005/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_908/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_908/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_1006/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1006/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_909/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_909/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_1007/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1007/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_910/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_910/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_1008/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1008/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_911/gamma/mRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_911/beta/mQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_1009/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1009/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_912/gamma/mRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_912/beta/mQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_1010/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1010/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_913/gamma/mRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_913/beta/mQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_1011/kernel/mSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1011/bias/mQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1001/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1001/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_904/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_904/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1002/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1002/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_905/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_905/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1003/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1003/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_906/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_906/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1004/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1004/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_907/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_907/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1005/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1005/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_908/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_908/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_1006/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1006/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_909/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_909/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_1007/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1007/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_910/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_910/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_1008/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1008/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_911/gamma/vRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_911/beta/vQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_1009/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1009/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_912/gamma/vRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_912/beta/vQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_1010/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1010/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_913/gamma/vRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_913/beta/vQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/dense_1011/kernel/vSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1011/bias/vQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

&serving_default_normalization_97_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Ê
StatefulPartitionedCallStatefulPartitionedCall&serving_default_normalization_97_inputConstConst_1dense_1001/kerneldense_1001/bias'batch_normalization_904/moving_variancebatch_normalization_904/gamma#batch_normalization_904/moving_meanbatch_normalization_904/betadense_1002/kerneldense_1002/bias'batch_normalization_905/moving_variancebatch_normalization_905/gamma#batch_normalization_905/moving_meanbatch_normalization_905/betadense_1003/kerneldense_1003/bias'batch_normalization_906/moving_variancebatch_normalization_906/gamma#batch_normalization_906/moving_meanbatch_normalization_906/betadense_1004/kerneldense_1004/bias'batch_normalization_907/moving_variancebatch_normalization_907/gamma#batch_normalization_907/moving_meanbatch_normalization_907/betadense_1005/kerneldense_1005/bias'batch_normalization_908/moving_variancebatch_normalization_908/gamma#batch_normalization_908/moving_meanbatch_normalization_908/betadense_1006/kerneldense_1006/bias'batch_normalization_909/moving_variancebatch_normalization_909/gamma#batch_normalization_909/moving_meanbatch_normalization_909/betadense_1007/kerneldense_1007/bias'batch_normalization_910/moving_variancebatch_normalization_910/gamma#batch_normalization_910/moving_meanbatch_normalization_910/betadense_1008/kerneldense_1008/bias'batch_normalization_911/moving_variancebatch_normalization_911/gamma#batch_normalization_911/moving_meanbatch_normalization_911/betadense_1009/kerneldense_1009/bias'batch_normalization_912/moving_variancebatch_normalization_912/gamma#batch_normalization_912/moving_meanbatch_normalization_912/betadense_1010/kerneldense_1010/bias'batch_normalization_913/moving_variancebatch_normalization_913/gamma#batch_normalization_913/moving_meanbatch_normalization_913/betadense_1011/kerneldense_1011/bias*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>?@*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1002852
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ó>
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp%dense_1001/kernel/Read/ReadVariableOp#dense_1001/bias/Read/ReadVariableOp1batch_normalization_904/gamma/Read/ReadVariableOp0batch_normalization_904/beta/Read/ReadVariableOp7batch_normalization_904/moving_mean/Read/ReadVariableOp;batch_normalization_904/moving_variance/Read/ReadVariableOp%dense_1002/kernel/Read/ReadVariableOp#dense_1002/bias/Read/ReadVariableOp1batch_normalization_905/gamma/Read/ReadVariableOp0batch_normalization_905/beta/Read/ReadVariableOp7batch_normalization_905/moving_mean/Read/ReadVariableOp;batch_normalization_905/moving_variance/Read/ReadVariableOp%dense_1003/kernel/Read/ReadVariableOp#dense_1003/bias/Read/ReadVariableOp1batch_normalization_906/gamma/Read/ReadVariableOp0batch_normalization_906/beta/Read/ReadVariableOp7batch_normalization_906/moving_mean/Read/ReadVariableOp;batch_normalization_906/moving_variance/Read/ReadVariableOp%dense_1004/kernel/Read/ReadVariableOp#dense_1004/bias/Read/ReadVariableOp1batch_normalization_907/gamma/Read/ReadVariableOp0batch_normalization_907/beta/Read/ReadVariableOp7batch_normalization_907/moving_mean/Read/ReadVariableOp;batch_normalization_907/moving_variance/Read/ReadVariableOp%dense_1005/kernel/Read/ReadVariableOp#dense_1005/bias/Read/ReadVariableOp1batch_normalization_908/gamma/Read/ReadVariableOp0batch_normalization_908/beta/Read/ReadVariableOp7batch_normalization_908/moving_mean/Read/ReadVariableOp;batch_normalization_908/moving_variance/Read/ReadVariableOp%dense_1006/kernel/Read/ReadVariableOp#dense_1006/bias/Read/ReadVariableOp1batch_normalization_909/gamma/Read/ReadVariableOp0batch_normalization_909/beta/Read/ReadVariableOp7batch_normalization_909/moving_mean/Read/ReadVariableOp;batch_normalization_909/moving_variance/Read/ReadVariableOp%dense_1007/kernel/Read/ReadVariableOp#dense_1007/bias/Read/ReadVariableOp1batch_normalization_910/gamma/Read/ReadVariableOp0batch_normalization_910/beta/Read/ReadVariableOp7batch_normalization_910/moving_mean/Read/ReadVariableOp;batch_normalization_910/moving_variance/Read/ReadVariableOp%dense_1008/kernel/Read/ReadVariableOp#dense_1008/bias/Read/ReadVariableOp1batch_normalization_911/gamma/Read/ReadVariableOp0batch_normalization_911/beta/Read/ReadVariableOp7batch_normalization_911/moving_mean/Read/ReadVariableOp;batch_normalization_911/moving_variance/Read/ReadVariableOp%dense_1009/kernel/Read/ReadVariableOp#dense_1009/bias/Read/ReadVariableOp1batch_normalization_912/gamma/Read/ReadVariableOp0batch_normalization_912/beta/Read/ReadVariableOp7batch_normalization_912/moving_mean/Read/ReadVariableOp;batch_normalization_912/moving_variance/Read/ReadVariableOp%dense_1010/kernel/Read/ReadVariableOp#dense_1010/bias/Read/ReadVariableOp1batch_normalization_913/gamma/Read/ReadVariableOp0batch_normalization_913/beta/Read/ReadVariableOp7batch_normalization_913/moving_mean/Read/ReadVariableOp;batch_normalization_913/moving_variance/Read/ReadVariableOp%dense_1011/kernel/Read/ReadVariableOp#dense_1011/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/dense_1001/kernel/m/Read/ReadVariableOp*Adam/dense_1001/bias/m/Read/ReadVariableOp8Adam/batch_normalization_904/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_904/beta/m/Read/ReadVariableOp,Adam/dense_1002/kernel/m/Read/ReadVariableOp*Adam/dense_1002/bias/m/Read/ReadVariableOp8Adam/batch_normalization_905/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_905/beta/m/Read/ReadVariableOp,Adam/dense_1003/kernel/m/Read/ReadVariableOp*Adam/dense_1003/bias/m/Read/ReadVariableOp8Adam/batch_normalization_906/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_906/beta/m/Read/ReadVariableOp,Adam/dense_1004/kernel/m/Read/ReadVariableOp*Adam/dense_1004/bias/m/Read/ReadVariableOp8Adam/batch_normalization_907/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_907/beta/m/Read/ReadVariableOp,Adam/dense_1005/kernel/m/Read/ReadVariableOp*Adam/dense_1005/bias/m/Read/ReadVariableOp8Adam/batch_normalization_908/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_908/beta/m/Read/ReadVariableOp,Adam/dense_1006/kernel/m/Read/ReadVariableOp*Adam/dense_1006/bias/m/Read/ReadVariableOp8Adam/batch_normalization_909/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_909/beta/m/Read/ReadVariableOp,Adam/dense_1007/kernel/m/Read/ReadVariableOp*Adam/dense_1007/bias/m/Read/ReadVariableOp8Adam/batch_normalization_910/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_910/beta/m/Read/ReadVariableOp,Adam/dense_1008/kernel/m/Read/ReadVariableOp*Adam/dense_1008/bias/m/Read/ReadVariableOp8Adam/batch_normalization_911/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_911/beta/m/Read/ReadVariableOp,Adam/dense_1009/kernel/m/Read/ReadVariableOp*Adam/dense_1009/bias/m/Read/ReadVariableOp8Adam/batch_normalization_912/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_912/beta/m/Read/ReadVariableOp,Adam/dense_1010/kernel/m/Read/ReadVariableOp*Adam/dense_1010/bias/m/Read/ReadVariableOp8Adam/batch_normalization_913/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_913/beta/m/Read/ReadVariableOp,Adam/dense_1011/kernel/m/Read/ReadVariableOp*Adam/dense_1011/bias/m/Read/ReadVariableOp,Adam/dense_1001/kernel/v/Read/ReadVariableOp*Adam/dense_1001/bias/v/Read/ReadVariableOp8Adam/batch_normalization_904/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_904/beta/v/Read/ReadVariableOp,Adam/dense_1002/kernel/v/Read/ReadVariableOp*Adam/dense_1002/bias/v/Read/ReadVariableOp8Adam/batch_normalization_905/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_905/beta/v/Read/ReadVariableOp,Adam/dense_1003/kernel/v/Read/ReadVariableOp*Adam/dense_1003/bias/v/Read/ReadVariableOp8Adam/batch_normalization_906/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_906/beta/v/Read/ReadVariableOp,Adam/dense_1004/kernel/v/Read/ReadVariableOp*Adam/dense_1004/bias/v/Read/ReadVariableOp8Adam/batch_normalization_907/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_907/beta/v/Read/ReadVariableOp,Adam/dense_1005/kernel/v/Read/ReadVariableOp*Adam/dense_1005/bias/v/Read/ReadVariableOp8Adam/batch_normalization_908/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_908/beta/v/Read/ReadVariableOp,Adam/dense_1006/kernel/v/Read/ReadVariableOp*Adam/dense_1006/bias/v/Read/ReadVariableOp8Adam/batch_normalization_909/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_909/beta/v/Read/ReadVariableOp,Adam/dense_1007/kernel/v/Read/ReadVariableOp*Adam/dense_1007/bias/v/Read/ReadVariableOp8Adam/batch_normalization_910/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_910/beta/v/Read/ReadVariableOp,Adam/dense_1008/kernel/v/Read/ReadVariableOp*Adam/dense_1008/bias/v/Read/ReadVariableOp8Adam/batch_normalization_911/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_911/beta/v/Read/ReadVariableOp,Adam/dense_1009/kernel/v/Read/ReadVariableOp*Adam/dense_1009/bias/v/Read/ReadVariableOp8Adam/batch_normalization_912/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_912/beta/v/Read/ReadVariableOp,Adam/dense_1010/kernel/v/Read/ReadVariableOp*Adam/dense_1010/bias/v/Read/ReadVariableOp8Adam/batch_normalization_913/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_913/beta/v/Read/ReadVariableOp,Adam/dense_1011/kernel/v/Read/ReadVariableOp*Adam/dense_1011/bias/v/Read/ReadVariableOpConst_2*«
Tin£
 2		*
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
 __inference__traced_save_1004498
°&
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_1001/kerneldense_1001/biasbatch_normalization_904/gammabatch_normalization_904/beta#batch_normalization_904/moving_mean'batch_normalization_904/moving_variancedense_1002/kerneldense_1002/biasbatch_normalization_905/gammabatch_normalization_905/beta#batch_normalization_905/moving_mean'batch_normalization_905/moving_variancedense_1003/kerneldense_1003/biasbatch_normalization_906/gammabatch_normalization_906/beta#batch_normalization_906/moving_mean'batch_normalization_906/moving_variancedense_1004/kerneldense_1004/biasbatch_normalization_907/gammabatch_normalization_907/beta#batch_normalization_907/moving_mean'batch_normalization_907/moving_variancedense_1005/kerneldense_1005/biasbatch_normalization_908/gammabatch_normalization_908/beta#batch_normalization_908/moving_mean'batch_normalization_908/moving_variancedense_1006/kerneldense_1006/biasbatch_normalization_909/gammabatch_normalization_909/beta#batch_normalization_909/moving_mean'batch_normalization_909/moving_variancedense_1007/kerneldense_1007/biasbatch_normalization_910/gammabatch_normalization_910/beta#batch_normalization_910/moving_mean'batch_normalization_910/moving_variancedense_1008/kerneldense_1008/biasbatch_normalization_911/gammabatch_normalization_911/beta#batch_normalization_911/moving_mean'batch_normalization_911/moving_variancedense_1009/kerneldense_1009/biasbatch_normalization_912/gammabatch_normalization_912/beta#batch_normalization_912/moving_mean'batch_normalization_912/moving_variancedense_1010/kerneldense_1010/biasbatch_normalization_913/gammabatch_normalization_913/beta#batch_normalization_913/moving_mean'batch_normalization_913/moving_variancedense_1011/kerneldense_1011/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount_1Adam/dense_1001/kernel/mAdam/dense_1001/bias/m$Adam/batch_normalization_904/gamma/m#Adam/batch_normalization_904/beta/mAdam/dense_1002/kernel/mAdam/dense_1002/bias/m$Adam/batch_normalization_905/gamma/m#Adam/batch_normalization_905/beta/mAdam/dense_1003/kernel/mAdam/dense_1003/bias/m$Adam/batch_normalization_906/gamma/m#Adam/batch_normalization_906/beta/mAdam/dense_1004/kernel/mAdam/dense_1004/bias/m$Adam/batch_normalization_907/gamma/m#Adam/batch_normalization_907/beta/mAdam/dense_1005/kernel/mAdam/dense_1005/bias/m$Adam/batch_normalization_908/gamma/m#Adam/batch_normalization_908/beta/mAdam/dense_1006/kernel/mAdam/dense_1006/bias/m$Adam/batch_normalization_909/gamma/m#Adam/batch_normalization_909/beta/mAdam/dense_1007/kernel/mAdam/dense_1007/bias/m$Adam/batch_normalization_910/gamma/m#Adam/batch_normalization_910/beta/mAdam/dense_1008/kernel/mAdam/dense_1008/bias/m$Adam/batch_normalization_911/gamma/m#Adam/batch_normalization_911/beta/mAdam/dense_1009/kernel/mAdam/dense_1009/bias/m$Adam/batch_normalization_912/gamma/m#Adam/batch_normalization_912/beta/mAdam/dense_1010/kernel/mAdam/dense_1010/bias/m$Adam/batch_normalization_913/gamma/m#Adam/batch_normalization_913/beta/mAdam/dense_1011/kernel/mAdam/dense_1011/bias/mAdam/dense_1001/kernel/vAdam/dense_1001/bias/v$Adam/batch_normalization_904/gamma/v#Adam/batch_normalization_904/beta/vAdam/dense_1002/kernel/vAdam/dense_1002/bias/v$Adam/batch_normalization_905/gamma/v#Adam/batch_normalization_905/beta/vAdam/dense_1003/kernel/vAdam/dense_1003/bias/v$Adam/batch_normalization_906/gamma/v#Adam/batch_normalization_906/beta/vAdam/dense_1004/kernel/vAdam/dense_1004/bias/v$Adam/batch_normalization_907/gamma/v#Adam/batch_normalization_907/beta/vAdam/dense_1005/kernel/vAdam/dense_1005/bias/v$Adam/batch_normalization_908/gamma/v#Adam/batch_normalization_908/beta/vAdam/dense_1006/kernel/vAdam/dense_1006/bias/v$Adam/batch_normalization_909/gamma/v#Adam/batch_normalization_909/beta/vAdam/dense_1007/kernel/vAdam/dense_1007/bias/v$Adam/batch_normalization_910/gamma/v#Adam/batch_normalization_910/beta/vAdam/dense_1008/kernel/vAdam/dense_1008/bias/v$Adam/batch_normalization_911/gamma/v#Adam/batch_normalization_911/beta/vAdam/dense_1009/kernel/vAdam/dense_1009/bias/v$Adam/batch_normalization_912/gamma/v#Adam/batch_normalization_912/beta/vAdam/dense_1010/kernel/vAdam/dense_1010/bias/v$Adam/batch_normalization_913/gamma/v#Adam/batch_normalization_913/beta/vAdam/dense_1011/kernel/vAdam/dense_1011/bias/v*ª
Tin¢
2*
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
#__inference__traced_restore_1004973¾Å&
æ
h
L__inference_leaky_re_lu_912_layer_call_and_return_conditional_losses_1000564

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1011_layer_call_and_return_conditional_losses_1000608

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_905_layer_call_and_return_conditional_losses_1003073

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
®
Ô
9__inference_batch_normalization_911_layer_call_fn_1003694

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_911_layer_call_and_return_conditional_losses_1000042o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
ó
/__inference_sequential_97_layer_call_fn_1001950

inputs
unknown
	unknown_0
	unknown_1:5
	unknown_2:5
	unknown_3:5
	unknown_4:5
	unknown_5:5
	unknown_6:5
	unknown_7:55
	unknown_8:5
	unknown_9:5

unknown_10:5

unknown_11:5

unknown_12:5

unknown_13:5

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:

unknown_55:

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:

unknown_61:

unknown_62:
identity¢StatefulPartitionedCall¸	
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
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>?@*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_97_layer_call_and_return_conditional_losses_1000615o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¸
_input_shapes¦
£:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
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
S__inference_batch_normalization_908_layer_call_and_return_conditional_losses_999843

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1004_layer_call_and_return_conditional_losses_1000384

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_906_layer_call_and_return_conditional_losses_1003182

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1007_layer_call_and_return_conditional_losses_1003572

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_913_layer_call_and_return_conditional_losses_1003979

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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
î§
­
J__inference_sequential_97_layer_call_and_return_conditional_losses_1001647
normalization_97_input
normalization_97_sub_y
normalization_97_sqrt_x$
dense_1001_1001491:5 
dense_1001_1001493:5-
batch_normalization_904_1001496:5-
batch_normalization_904_1001498:5-
batch_normalization_904_1001500:5-
batch_normalization_904_1001502:5$
dense_1002_1001506:55 
dense_1002_1001508:5-
batch_normalization_905_1001511:5-
batch_normalization_905_1001513:5-
batch_normalization_905_1001515:5-
batch_normalization_905_1001517:5$
dense_1003_1001521:5 
dense_1003_1001523:-
batch_normalization_906_1001526:-
batch_normalization_906_1001528:-
batch_normalization_906_1001530:-
batch_normalization_906_1001532:$
dense_1004_1001536: 
dense_1004_1001538:-
batch_normalization_907_1001541:-
batch_normalization_907_1001543:-
batch_normalization_907_1001545:-
batch_normalization_907_1001547:$
dense_1005_1001551: 
dense_1005_1001553:-
batch_normalization_908_1001556:-
batch_normalization_908_1001558:-
batch_normalization_908_1001560:-
batch_normalization_908_1001562:$
dense_1006_1001566: 
dense_1006_1001568:-
batch_normalization_909_1001571:-
batch_normalization_909_1001573:-
batch_normalization_909_1001575:-
batch_normalization_909_1001577:$
dense_1007_1001581: 
dense_1007_1001583:-
batch_normalization_910_1001586:-
batch_normalization_910_1001588:-
batch_normalization_910_1001590:-
batch_normalization_910_1001592:$
dense_1008_1001596: 
dense_1008_1001598:-
batch_normalization_911_1001601:-
batch_normalization_911_1001603:-
batch_normalization_911_1001605:-
batch_normalization_911_1001607:$
dense_1009_1001611: 
dense_1009_1001613:-
batch_normalization_912_1001616:-
batch_normalization_912_1001618:-
batch_normalization_912_1001620:-
batch_normalization_912_1001622:$
dense_1010_1001626: 
dense_1010_1001628:-
batch_normalization_913_1001631:-
batch_normalization_913_1001633:-
batch_normalization_913_1001635:-
batch_normalization_913_1001637:$
dense_1011_1001641: 
dense_1011_1001643:
identity¢/batch_normalization_904/StatefulPartitionedCall¢/batch_normalization_905/StatefulPartitionedCall¢/batch_normalization_906/StatefulPartitionedCall¢/batch_normalization_907/StatefulPartitionedCall¢/batch_normalization_908/StatefulPartitionedCall¢/batch_normalization_909/StatefulPartitionedCall¢/batch_normalization_910/StatefulPartitionedCall¢/batch_normalization_911/StatefulPartitionedCall¢/batch_normalization_912/StatefulPartitionedCall¢/batch_normalization_913/StatefulPartitionedCall¢"dense_1001/StatefulPartitionedCall¢"dense_1002/StatefulPartitionedCall¢"dense_1003/StatefulPartitionedCall¢"dense_1004/StatefulPartitionedCall¢"dense_1005/StatefulPartitionedCall¢"dense_1006/StatefulPartitionedCall¢"dense_1007/StatefulPartitionedCall¢"dense_1008/StatefulPartitionedCall¢"dense_1009/StatefulPartitionedCall¢"dense_1010/StatefulPartitionedCall¢"dense_1011/StatefulPartitionedCall}
normalization_97/subSubnormalization_97_inputnormalization_97_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_97/SqrtSqrtnormalization_97_sqrt_x*
T0*
_output_shapes

:_
normalization_97/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_97/MaximumMaximumnormalization_97/Sqrt:y:0#normalization_97/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_97/truedivRealDivnormalization_97/sub:z:0normalization_97/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"dense_1001/StatefulPartitionedCallStatefulPartitionedCallnormalization_97/truediv:z:0dense_1001_1001491dense_1001_1001493*
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
GPU 2J 8 *P
fKRI
G__inference_dense_1001_layer_call_and_return_conditional_losses_1000288
/batch_normalization_904/StatefulPartitionedCallStatefulPartitionedCall+dense_1001/StatefulPartitionedCall:output:0batch_normalization_904_1001496batch_normalization_904_1001498batch_normalization_904_1001500batch_normalization_904_1001502*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_904_layer_call_and_return_conditional_losses_999468ù
leaky_re_lu_904/PartitionedCallPartitionedCall8batch_normalization_904/StatefulPartitionedCall:output:0*
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
L__inference_leaky_re_lu_904_layer_call_and_return_conditional_losses_1000308
"dense_1002/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_904/PartitionedCall:output:0dense_1002_1001506dense_1002_1001508*
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
GPU 2J 8 *P
fKRI
G__inference_dense_1002_layer_call_and_return_conditional_losses_1000320
/batch_normalization_905/StatefulPartitionedCallStatefulPartitionedCall+dense_1002/StatefulPartitionedCall:output:0batch_normalization_905_1001511batch_normalization_905_1001513batch_normalization_905_1001515batch_normalization_905_1001517*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_905_layer_call_and_return_conditional_losses_999550ù
leaky_re_lu_905/PartitionedCallPartitionedCall8batch_normalization_905/StatefulPartitionedCall:output:0*
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
L__inference_leaky_re_lu_905_layer_call_and_return_conditional_losses_1000340
"dense_1003/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_905/PartitionedCall:output:0dense_1003_1001521dense_1003_1001523*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1003_layer_call_and_return_conditional_losses_1000352
/batch_normalization_906/StatefulPartitionedCallStatefulPartitionedCall+dense_1003/StatefulPartitionedCall:output:0batch_normalization_906_1001526batch_normalization_906_1001528batch_normalization_906_1001530batch_normalization_906_1001532*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_906_layer_call_and_return_conditional_losses_999632ù
leaky_re_lu_906/PartitionedCallPartitionedCall8batch_normalization_906/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_906_layer_call_and_return_conditional_losses_1000372
"dense_1004/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_906/PartitionedCall:output:0dense_1004_1001536dense_1004_1001538*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1004_layer_call_and_return_conditional_losses_1000384
/batch_normalization_907/StatefulPartitionedCallStatefulPartitionedCall+dense_1004/StatefulPartitionedCall:output:0batch_normalization_907_1001541batch_normalization_907_1001543batch_normalization_907_1001545batch_normalization_907_1001547*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_907_layer_call_and_return_conditional_losses_999714ù
leaky_re_lu_907/PartitionedCallPartitionedCall8batch_normalization_907/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_907_layer_call_and_return_conditional_losses_1000404
"dense_1005/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_907/PartitionedCall:output:0dense_1005_1001551dense_1005_1001553*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1005_layer_call_and_return_conditional_losses_1000416
/batch_normalization_908/StatefulPartitionedCallStatefulPartitionedCall+dense_1005/StatefulPartitionedCall:output:0batch_normalization_908_1001556batch_normalization_908_1001558batch_normalization_908_1001560batch_normalization_908_1001562*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_908_layer_call_and_return_conditional_losses_999796ù
leaky_re_lu_908/PartitionedCallPartitionedCall8batch_normalization_908/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_908_layer_call_and_return_conditional_losses_1000436
"dense_1006/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_908/PartitionedCall:output:0dense_1006_1001566dense_1006_1001568*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1006_layer_call_and_return_conditional_losses_1000448
/batch_normalization_909/StatefulPartitionedCallStatefulPartitionedCall+dense_1006/StatefulPartitionedCall:output:0batch_normalization_909_1001571batch_normalization_909_1001573batch_normalization_909_1001575batch_normalization_909_1001577*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_909_layer_call_and_return_conditional_losses_999878ù
leaky_re_lu_909/PartitionedCallPartitionedCall8batch_normalization_909/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_909_layer_call_and_return_conditional_losses_1000468
"dense_1007/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_909/PartitionedCall:output:0dense_1007_1001581dense_1007_1001583*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1007_layer_call_and_return_conditional_losses_1000480
/batch_normalization_910/StatefulPartitionedCallStatefulPartitionedCall+dense_1007/StatefulPartitionedCall:output:0batch_normalization_910_1001586batch_normalization_910_1001588batch_normalization_910_1001590batch_normalization_910_1001592*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_910_layer_call_and_return_conditional_losses_999960ù
leaky_re_lu_910/PartitionedCallPartitionedCall8batch_normalization_910/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_910_layer_call_and_return_conditional_losses_1000500
"dense_1008/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_910/PartitionedCall:output:0dense_1008_1001596dense_1008_1001598*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1008_layer_call_and_return_conditional_losses_1000512
/batch_normalization_911/StatefulPartitionedCallStatefulPartitionedCall+dense_1008/StatefulPartitionedCall:output:0batch_normalization_911_1001601batch_normalization_911_1001603batch_normalization_911_1001605batch_normalization_911_1001607*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_911_layer_call_and_return_conditional_losses_1000042ù
leaky_re_lu_911/PartitionedCallPartitionedCall8batch_normalization_911/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_911_layer_call_and_return_conditional_losses_1000532
"dense_1009/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_911/PartitionedCall:output:0dense_1009_1001611dense_1009_1001613*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1009_layer_call_and_return_conditional_losses_1000544
/batch_normalization_912/StatefulPartitionedCallStatefulPartitionedCall+dense_1009/StatefulPartitionedCall:output:0batch_normalization_912_1001616batch_normalization_912_1001618batch_normalization_912_1001620batch_normalization_912_1001622*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_912_layer_call_and_return_conditional_losses_1000124ù
leaky_re_lu_912/PartitionedCallPartitionedCall8batch_normalization_912/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_912_layer_call_and_return_conditional_losses_1000564
"dense_1010/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_912/PartitionedCall:output:0dense_1010_1001626dense_1010_1001628*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1010_layer_call_and_return_conditional_losses_1000576
/batch_normalization_913/StatefulPartitionedCallStatefulPartitionedCall+dense_1010/StatefulPartitionedCall:output:0batch_normalization_913_1001631batch_normalization_913_1001633batch_normalization_913_1001635batch_normalization_913_1001637*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_913_layer_call_and_return_conditional_losses_1000206ù
leaky_re_lu_913/PartitionedCallPartitionedCall8batch_normalization_913/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_913_layer_call_and_return_conditional_losses_1000596
"dense_1011/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_913/PartitionedCall:output:0dense_1011_1001641dense_1011_1001643*
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
GPU 2J 8 *P
fKRI
G__inference_dense_1011_layer_call_and_return_conditional_losses_1000608z
IdentityIdentity+dense_1011/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp0^batch_normalization_904/StatefulPartitionedCall0^batch_normalization_905/StatefulPartitionedCall0^batch_normalization_906/StatefulPartitionedCall0^batch_normalization_907/StatefulPartitionedCall0^batch_normalization_908/StatefulPartitionedCall0^batch_normalization_909/StatefulPartitionedCall0^batch_normalization_910/StatefulPartitionedCall0^batch_normalization_911/StatefulPartitionedCall0^batch_normalization_912/StatefulPartitionedCall0^batch_normalization_913/StatefulPartitionedCall#^dense_1001/StatefulPartitionedCall#^dense_1002/StatefulPartitionedCall#^dense_1003/StatefulPartitionedCall#^dense_1004/StatefulPartitionedCall#^dense_1005/StatefulPartitionedCall#^dense_1006/StatefulPartitionedCall#^dense_1007/StatefulPartitionedCall#^dense_1008/StatefulPartitionedCall#^dense_1009/StatefulPartitionedCall#^dense_1010/StatefulPartitionedCall#^dense_1011/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¸
_input_shapes¦
£:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_904/StatefulPartitionedCall/batch_normalization_904/StatefulPartitionedCall2b
/batch_normalization_905/StatefulPartitionedCall/batch_normalization_905/StatefulPartitionedCall2b
/batch_normalization_906/StatefulPartitionedCall/batch_normalization_906/StatefulPartitionedCall2b
/batch_normalization_907/StatefulPartitionedCall/batch_normalization_907/StatefulPartitionedCall2b
/batch_normalization_908/StatefulPartitionedCall/batch_normalization_908/StatefulPartitionedCall2b
/batch_normalization_909/StatefulPartitionedCall/batch_normalization_909/StatefulPartitionedCall2b
/batch_normalization_910/StatefulPartitionedCall/batch_normalization_910/StatefulPartitionedCall2b
/batch_normalization_911/StatefulPartitionedCall/batch_normalization_911/StatefulPartitionedCall2b
/batch_normalization_912/StatefulPartitionedCall/batch_normalization_912/StatefulPartitionedCall2b
/batch_normalization_913/StatefulPartitionedCall/batch_normalization_913/StatefulPartitionedCall2H
"dense_1001/StatefulPartitionedCall"dense_1001/StatefulPartitionedCall2H
"dense_1002/StatefulPartitionedCall"dense_1002/StatefulPartitionedCall2H
"dense_1003/StatefulPartitionedCall"dense_1003/StatefulPartitionedCall2H
"dense_1004/StatefulPartitionedCall"dense_1004/StatefulPartitionedCall2H
"dense_1005/StatefulPartitionedCall"dense_1005/StatefulPartitionedCall2H
"dense_1006/StatefulPartitionedCall"dense_1006/StatefulPartitionedCall2H
"dense_1007/StatefulPartitionedCall"dense_1007/StatefulPartitionedCall2H
"dense_1008/StatefulPartitionedCall"dense_1008/StatefulPartitionedCall2H
"dense_1009/StatefulPartitionedCall"dense_1009/StatefulPartitionedCall2H
"dense_1010/StatefulPartitionedCall"dense_1010/StatefulPartitionedCall2H
"dense_1011/StatefulPartitionedCall"dense_1011/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_97_input:$ 

_output_shapes

::$ 

_output_shapes

:
­
M
1__inference_leaky_re_lu_904_layer_call_fn_1003003

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
L__inference_leaky_re_lu_904_layer_call_and_return_conditional_losses_1000308`
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
%
ì
S__inference_batch_normalization_905_layer_call_and_return_conditional_losses_999597

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
È

,__inference_dense_1003_layer_call_fn_1003126

inputs
unknown:5
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1003_layer_call_and_return_conditional_losses_1000352o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
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
L__inference_leaky_re_lu_905_layer_call_and_return_conditional_losses_1003117

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
È

,__inference_dense_1010_layer_call_fn_1003889

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1010_layer_call_and_return_conditional_losses_1000576o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_904_layer_call_and_return_conditional_losses_1002964

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
1__inference_leaky_re_lu_912_layer_call_fn_1003875

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_912_layer_call_and_return_conditional_losses_1000564`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö·
G
!__inference__wrapped_model_999444
normalization_97_input(
$sequential_97_normalization_97_sub_y)
%sequential_97_normalization_97_sqrt_xI
7sequential_97_dense_1001_matmul_readvariableop_resource:5F
8sequential_97_dense_1001_biasadd_readvariableop_resource:5U
Gsequential_97_batch_normalization_904_batchnorm_readvariableop_resource:5Y
Ksequential_97_batch_normalization_904_batchnorm_mul_readvariableop_resource:5W
Isequential_97_batch_normalization_904_batchnorm_readvariableop_1_resource:5W
Isequential_97_batch_normalization_904_batchnorm_readvariableop_2_resource:5I
7sequential_97_dense_1002_matmul_readvariableop_resource:55F
8sequential_97_dense_1002_biasadd_readvariableop_resource:5U
Gsequential_97_batch_normalization_905_batchnorm_readvariableop_resource:5Y
Ksequential_97_batch_normalization_905_batchnorm_mul_readvariableop_resource:5W
Isequential_97_batch_normalization_905_batchnorm_readvariableop_1_resource:5W
Isequential_97_batch_normalization_905_batchnorm_readvariableop_2_resource:5I
7sequential_97_dense_1003_matmul_readvariableop_resource:5F
8sequential_97_dense_1003_biasadd_readvariableop_resource:U
Gsequential_97_batch_normalization_906_batchnorm_readvariableop_resource:Y
Ksequential_97_batch_normalization_906_batchnorm_mul_readvariableop_resource:W
Isequential_97_batch_normalization_906_batchnorm_readvariableop_1_resource:W
Isequential_97_batch_normalization_906_batchnorm_readvariableop_2_resource:I
7sequential_97_dense_1004_matmul_readvariableop_resource:F
8sequential_97_dense_1004_biasadd_readvariableop_resource:U
Gsequential_97_batch_normalization_907_batchnorm_readvariableop_resource:Y
Ksequential_97_batch_normalization_907_batchnorm_mul_readvariableop_resource:W
Isequential_97_batch_normalization_907_batchnorm_readvariableop_1_resource:W
Isequential_97_batch_normalization_907_batchnorm_readvariableop_2_resource:I
7sequential_97_dense_1005_matmul_readvariableop_resource:F
8sequential_97_dense_1005_biasadd_readvariableop_resource:U
Gsequential_97_batch_normalization_908_batchnorm_readvariableop_resource:Y
Ksequential_97_batch_normalization_908_batchnorm_mul_readvariableop_resource:W
Isequential_97_batch_normalization_908_batchnorm_readvariableop_1_resource:W
Isequential_97_batch_normalization_908_batchnorm_readvariableop_2_resource:I
7sequential_97_dense_1006_matmul_readvariableop_resource:F
8sequential_97_dense_1006_biasadd_readvariableop_resource:U
Gsequential_97_batch_normalization_909_batchnorm_readvariableop_resource:Y
Ksequential_97_batch_normalization_909_batchnorm_mul_readvariableop_resource:W
Isequential_97_batch_normalization_909_batchnorm_readvariableop_1_resource:W
Isequential_97_batch_normalization_909_batchnorm_readvariableop_2_resource:I
7sequential_97_dense_1007_matmul_readvariableop_resource:F
8sequential_97_dense_1007_biasadd_readvariableop_resource:U
Gsequential_97_batch_normalization_910_batchnorm_readvariableop_resource:Y
Ksequential_97_batch_normalization_910_batchnorm_mul_readvariableop_resource:W
Isequential_97_batch_normalization_910_batchnorm_readvariableop_1_resource:W
Isequential_97_batch_normalization_910_batchnorm_readvariableop_2_resource:I
7sequential_97_dense_1008_matmul_readvariableop_resource:F
8sequential_97_dense_1008_biasadd_readvariableop_resource:U
Gsequential_97_batch_normalization_911_batchnorm_readvariableop_resource:Y
Ksequential_97_batch_normalization_911_batchnorm_mul_readvariableop_resource:W
Isequential_97_batch_normalization_911_batchnorm_readvariableop_1_resource:W
Isequential_97_batch_normalization_911_batchnorm_readvariableop_2_resource:I
7sequential_97_dense_1009_matmul_readvariableop_resource:F
8sequential_97_dense_1009_biasadd_readvariableop_resource:U
Gsequential_97_batch_normalization_912_batchnorm_readvariableop_resource:Y
Ksequential_97_batch_normalization_912_batchnorm_mul_readvariableop_resource:W
Isequential_97_batch_normalization_912_batchnorm_readvariableop_1_resource:W
Isequential_97_batch_normalization_912_batchnorm_readvariableop_2_resource:I
7sequential_97_dense_1010_matmul_readvariableop_resource:F
8sequential_97_dense_1010_biasadd_readvariableop_resource:U
Gsequential_97_batch_normalization_913_batchnorm_readvariableop_resource:Y
Ksequential_97_batch_normalization_913_batchnorm_mul_readvariableop_resource:W
Isequential_97_batch_normalization_913_batchnorm_readvariableop_1_resource:W
Isequential_97_batch_normalization_913_batchnorm_readvariableop_2_resource:I
7sequential_97_dense_1011_matmul_readvariableop_resource:F
8sequential_97_dense_1011_biasadd_readvariableop_resource:
identity¢>sequential_97/batch_normalization_904/batchnorm/ReadVariableOp¢@sequential_97/batch_normalization_904/batchnorm/ReadVariableOp_1¢@sequential_97/batch_normalization_904/batchnorm/ReadVariableOp_2¢Bsequential_97/batch_normalization_904/batchnorm/mul/ReadVariableOp¢>sequential_97/batch_normalization_905/batchnorm/ReadVariableOp¢@sequential_97/batch_normalization_905/batchnorm/ReadVariableOp_1¢@sequential_97/batch_normalization_905/batchnorm/ReadVariableOp_2¢Bsequential_97/batch_normalization_905/batchnorm/mul/ReadVariableOp¢>sequential_97/batch_normalization_906/batchnorm/ReadVariableOp¢@sequential_97/batch_normalization_906/batchnorm/ReadVariableOp_1¢@sequential_97/batch_normalization_906/batchnorm/ReadVariableOp_2¢Bsequential_97/batch_normalization_906/batchnorm/mul/ReadVariableOp¢>sequential_97/batch_normalization_907/batchnorm/ReadVariableOp¢@sequential_97/batch_normalization_907/batchnorm/ReadVariableOp_1¢@sequential_97/batch_normalization_907/batchnorm/ReadVariableOp_2¢Bsequential_97/batch_normalization_907/batchnorm/mul/ReadVariableOp¢>sequential_97/batch_normalization_908/batchnorm/ReadVariableOp¢@sequential_97/batch_normalization_908/batchnorm/ReadVariableOp_1¢@sequential_97/batch_normalization_908/batchnorm/ReadVariableOp_2¢Bsequential_97/batch_normalization_908/batchnorm/mul/ReadVariableOp¢>sequential_97/batch_normalization_909/batchnorm/ReadVariableOp¢@sequential_97/batch_normalization_909/batchnorm/ReadVariableOp_1¢@sequential_97/batch_normalization_909/batchnorm/ReadVariableOp_2¢Bsequential_97/batch_normalization_909/batchnorm/mul/ReadVariableOp¢>sequential_97/batch_normalization_910/batchnorm/ReadVariableOp¢@sequential_97/batch_normalization_910/batchnorm/ReadVariableOp_1¢@sequential_97/batch_normalization_910/batchnorm/ReadVariableOp_2¢Bsequential_97/batch_normalization_910/batchnorm/mul/ReadVariableOp¢>sequential_97/batch_normalization_911/batchnorm/ReadVariableOp¢@sequential_97/batch_normalization_911/batchnorm/ReadVariableOp_1¢@sequential_97/batch_normalization_911/batchnorm/ReadVariableOp_2¢Bsequential_97/batch_normalization_911/batchnorm/mul/ReadVariableOp¢>sequential_97/batch_normalization_912/batchnorm/ReadVariableOp¢@sequential_97/batch_normalization_912/batchnorm/ReadVariableOp_1¢@sequential_97/batch_normalization_912/batchnorm/ReadVariableOp_2¢Bsequential_97/batch_normalization_912/batchnorm/mul/ReadVariableOp¢>sequential_97/batch_normalization_913/batchnorm/ReadVariableOp¢@sequential_97/batch_normalization_913/batchnorm/ReadVariableOp_1¢@sequential_97/batch_normalization_913/batchnorm/ReadVariableOp_2¢Bsequential_97/batch_normalization_913/batchnorm/mul/ReadVariableOp¢/sequential_97/dense_1001/BiasAdd/ReadVariableOp¢.sequential_97/dense_1001/MatMul/ReadVariableOp¢/sequential_97/dense_1002/BiasAdd/ReadVariableOp¢.sequential_97/dense_1002/MatMul/ReadVariableOp¢/sequential_97/dense_1003/BiasAdd/ReadVariableOp¢.sequential_97/dense_1003/MatMul/ReadVariableOp¢/sequential_97/dense_1004/BiasAdd/ReadVariableOp¢.sequential_97/dense_1004/MatMul/ReadVariableOp¢/sequential_97/dense_1005/BiasAdd/ReadVariableOp¢.sequential_97/dense_1005/MatMul/ReadVariableOp¢/sequential_97/dense_1006/BiasAdd/ReadVariableOp¢.sequential_97/dense_1006/MatMul/ReadVariableOp¢/sequential_97/dense_1007/BiasAdd/ReadVariableOp¢.sequential_97/dense_1007/MatMul/ReadVariableOp¢/sequential_97/dense_1008/BiasAdd/ReadVariableOp¢.sequential_97/dense_1008/MatMul/ReadVariableOp¢/sequential_97/dense_1009/BiasAdd/ReadVariableOp¢.sequential_97/dense_1009/MatMul/ReadVariableOp¢/sequential_97/dense_1010/BiasAdd/ReadVariableOp¢.sequential_97/dense_1010/MatMul/ReadVariableOp¢/sequential_97/dense_1011/BiasAdd/ReadVariableOp¢.sequential_97/dense_1011/MatMul/ReadVariableOp
"sequential_97/normalization_97/subSubnormalization_97_input$sequential_97_normalization_97_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
#sequential_97/normalization_97/SqrtSqrt%sequential_97_normalization_97_sqrt_x*
T0*
_output_shapes

:m
(sequential_97/normalization_97/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¶
&sequential_97/normalization_97/MaximumMaximum'sequential_97/normalization_97/Sqrt:y:01sequential_97/normalization_97/Maximum/y:output:0*
T0*
_output_shapes

:·
&sequential_97/normalization_97/truedivRealDiv&sequential_97/normalization_97/sub:z:0*sequential_97/normalization_97/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
.sequential_97/dense_1001/MatMul/ReadVariableOpReadVariableOp7sequential_97_dense_1001_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0¿
sequential_97/dense_1001/MatMulMatMul*sequential_97/normalization_97/truediv:z:06sequential_97/dense_1001/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¤
/sequential_97/dense_1001/BiasAdd/ReadVariableOpReadVariableOp8sequential_97_dense_1001_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0Á
 sequential_97/dense_1001/BiasAddBiasAdd)sequential_97/dense_1001/MatMul:product:07sequential_97/dense_1001/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5Â
>sequential_97/batch_normalization_904/batchnorm/ReadVariableOpReadVariableOpGsequential_97_batch_normalization_904_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0z
5sequential_97/batch_normalization_904/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_97/batch_normalization_904/batchnorm/addAddV2Fsequential_97/batch_normalization_904/batchnorm/ReadVariableOp:value:0>sequential_97/batch_normalization_904/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
5sequential_97/batch_normalization_904/batchnorm/RsqrtRsqrt7sequential_97/batch_normalization_904/batchnorm/add:z:0*
T0*
_output_shapes
:5Ê
Bsequential_97/batch_normalization_904/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_97_batch_normalization_904_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0æ
3sequential_97/batch_normalization_904/batchnorm/mulMul9sequential_97/batch_normalization_904/batchnorm/Rsqrt:y:0Jsequential_97/batch_normalization_904/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5Ò
5sequential_97/batch_normalization_904/batchnorm/mul_1Mul)sequential_97/dense_1001/BiasAdd:output:07sequential_97/batch_normalization_904/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5Æ
@sequential_97/batch_normalization_904/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_97_batch_normalization_904_batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0ä
5sequential_97/batch_normalization_904/batchnorm/mul_2MulHsequential_97/batch_normalization_904/batchnorm/ReadVariableOp_1:value:07sequential_97/batch_normalization_904/batchnorm/mul:z:0*
T0*
_output_shapes
:5Æ
@sequential_97/batch_normalization_904/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_97_batch_normalization_904_batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0ä
3sequential_97/batch_normalization_904/batchnorm/subSubHsequential_97/batch_normalization_904/batchnorm/ReadVariableOp_2:value:09sequential_97/batch_normalization_904/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5ä
5sequential_97/batch_normalization_904/batchnorm/add_1AddV29sequential_97/batch_normalization_904/batchnorm/mul_1:z:07sequential_97/batch_normalization_904/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¨
'sequential_97/leaky_re_lu_904/LeakyRelu	LeakyRelu9sequential_97/batch_normalization_904/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>¦
.sequential_97/dense_1002/MatMul/ReadVariableOpReadVariableOp7sequential_97_dense_1002_matmul_readvariableop_resource*
_output_shapes

:55*
dtype0Ê
sequential_97/dense_1002/MatMulMatMul5sequential_97/leaky_re_lu_904/LeakyRelu:activations:06sequential_97/dense_1002/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¤
/sequential_97/dense_1002/BiasAdd/ReadVariableOpReadVariableOp8sequential_97_dense_1002_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0Á
 sequential_97/dense_1002/BiasAddBiasAdd)sequential_97/dense_1002/MatMul:product:07sequential_97/dense_1002/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5Â
>sequential_97/batch_normalization_905/batchnorm/ReadVariableOpReadVariableOpGsequential_97_batch_normalization_905_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0z
5sequential_97/batch_normalization_905/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_97/batch_normalization_905/batchnorm/addAddV2Fsequential_97/batch_normalization_905/batchnorm/ReadVariableOp:value:0>sequential_97/batch_normalization_905/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
5sequential_97/batch_normalization_905/batchnorm/RsqrtRsqrt7sequential_97/batch_normalization_905/batchnorm/add:z:0*
T0*
_output_shapes
:5Ê
Bsequential_97/batch_normalization_905/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_97_batch_normalization_905_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0æ
3sequential_97/batch_normalization_905/batchnorm/mulMul9sequential_97/batch_normalization_905/batchnorm/Rsqrt:y:0Jsequential_97/batch_normalization_905/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5Ò
5sequential_97/batch_normalization_905/batchnorm/mul_1Mul)sequential_97/dense_1002/BiasAdd:output:07sequential_97/batch_normalization_905/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5Æ
@sequential_97/batch_normalization_905/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_97_batch_normalization_905_batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0ä
5sequential_97/batch_normalization_905/batchnorm/mul_2MulHsequential_97/batch_normalization_905/batchnorm/ReadVariableOp_1:value:07sequential_97/batch_normalization_905/batchnorm/mul:z:0*
T0*
_output_shapes
:5Æ
@sequential_97/batch_normalization_905/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_97_batch_normalization_905_batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0ä
3sequential_97/batch_normalization_905/batchnorm/subSubHsequential_97/batch_normalization_905/batchnorm/ReadVariableOp_2:value:09sequential_97/batch_normalization_905/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5ä
5sequential_97/batch_normalization_905/batchnorm/add_1AddV29sequential_97/batch_normalization_905/batchnorm/mul_1:z:07sequential_97/batch_normalization_905/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¨
'sequential_97/leaky_re_lu_905/LeakyRelu	LeakyRelu9sequential_97/batch_normalization_905/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>¦
.sequential_97/dense_1003/MatMul/ReadVariableOpReadVariableOp7sequential_97_dense_1003_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0Ê
sequential_97/dense_1003/MatMulMatMul5sequential_97/leaky_re_lu_905/LeakyRelu:activations:06sequential_97/dense_1003/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/sequential_97/dense_1003/BiasAdd/ReadVariableOpReadVariableOp8sequential_97_dense_1003_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 sequential_97/dense_1003/BiasAddBiasAdd)sequential_97/dense_1003/MatMul:product:07sequential_97/dense_1003/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_97/batch_normalization_906/batchnorm/ReadVariableOpReadVariableOpGsequential_97_batch_normalization_906_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_97/batch_normalization_906/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_97/batch_normalization_906/batchnorm/addAddV2Fsequential_97/batch_normalization_906/batchnorm/ReadVariableOp:value:0>sequential_97/batch_normalization_906/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_97/batch_normalization_906/batchnorm/RsqrtRsqrt7sequential_97/batch_normalization_906/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_97/batch_normalization_906/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_97_batch_normalization_906_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_97/batch_normalization_906/batchnorm/mulMul9sequential_97/batch_normalization_906/batchnorm/Rsqrt:y:0Jsequential_97/batch_normalization_906/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ò
5sequential_97/batch_normalization_906/batchnorm/mul_1Mul)sequential_97/dense_1003/BiasAdd:output:07sequential_97/batch_normalization_906/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_97/batch_normalization_906/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_97_batch_normalization_906_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_97/batch_normalization_906/batchnorm/mul_2MulHsequential_97/batch_normalization_906/batchnorm/ReadVariableOp_1:value:07sequential_97/batch_normalization_906/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_97/batch_normalization_906/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_97_batch_normalization_906_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_97/batch_normalization_906/batchnorm/subSubHsequential_97/batch_normalization_906/batchnorm/ReadVariableOp_2:value:09sequential_97/batch_normalization_906/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_97/batch_normalization_906/batchnorm/add_1AddV29sequential_97/batch_normalization_906/batchnorm/mul_1:z:07sequential_97/batch_normalization_906/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_97/leaky_re_lu_906/LeakyRelu	LeakyRelu9sequential_97/batch_normalization_906/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¦
.sequential_97/dense_1004/MatMul/ReadVariableOpReadVariableOp7sequential_97_dense_1004_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
sequential_97/dense_1004/MatMulMatMul5sequential_97/leaky_re_lu_906/LeakyRelu:activations:06sequential_97/dense_1004/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/sequential_97/dense_1004/BiasAdd/ReadVariableOpReadVariableOp8sequential_97_dense_1004_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 sequential_97/dense_1004/BiasAddBiasAdd)sequential_97/dense_1004/MatMul:product:07sequential_97/dense_1004/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_97/batch_normalization_907/batchnorm/ReadVariableOpReadVariableOpGsequential_97_batch_normalization_907_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_97/batch_normalization_907/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_97/batch_normalization_907/batchnorm/addAddV2Fsequential_97/batch_normalization_907/batchnorm/ReadVariableOp:value:0>sequential_97/batch_normalization_907/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_97/batch_normalization_907/batchnorm/RsqrtRsqrt7sequential_97/batch_normalization_907/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_97/batch_normalization_907/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_97_batch_normalization_907_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_97/batch_normalization_907/batchnorm/mulMul9sequential_97/batch_normalization_907/batchnorm/Rsqrt:y:0Jsequential_97/batch_normalization_907/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ò
5sequential_97/batch_normalization_907/batchnorm/mul_1Mul)sequential_97/dense_1004/BiasAdd:output:07sequential_97/batch_normalization_907/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_97/batch_normalization_907/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_97_batch_normalization_907_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_97/batch_normalization_907/batchnorm/mul_2MulHsequential_97/batch_normalization_907/batchnorm/ReadVariableOp_1:value:07sequential_97/batch_normalization_907/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_97/batch_normalization_907/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_97_batch_normalization_907_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_97/batch_normalization_907/batchnorm/subSubHsequential_97/batch_normalization_907/batchnorm/ReadVariableOp_2:value:09sequential_97/batch_normalization_907/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_97/batch_normalization_907/batchnorm/add_1AddV29sequential_97/batch_normalization_907/batchnorm/mul_1:z:07sequential_97/batch_normalization_907/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_97/leaky_re_lu_907/LeakyRelu	LeakyRelu9sequential_97/batch_normalization_907/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¦
.sequential_97/dense_1005/MatMul/ReadVariableOpReadVariableOp7sequential_97_dense_1005_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
sequential_97/dense_1005/MatMulMatMul5sequential_97/leaky_re_lu_907/LeakyRelu:activations:06sequential_97/dense_1005/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/sequential_97/dense_1005/BiasAdd/ReadVariableOpReadVariableOp8sequential_97_dense_1005_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 sequential_97/dense_1005/BiasAddBiasAdd)sequential_97/dense_1005/MatMul:product:07sequential_97/dense_1005/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_97/batch_normalization_908/batchnorm/ReadVariableOpReadVariableOpGsequential_97_batch_normalization_908_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_97/batch_normalization_908/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_97/batch_normalization_908/batchnorm/addAddV2Fsequential_97/batch_normalization_908/batchnorm/ReadVariableOp:value:0>sequential_97/batch_normalization_908/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_97/batch_normalization_908/batchnorm/RsqrtRsqrt7sequential_97/batch_normalization_908/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_97/batch_normalization_908/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_97_batch_normalization_908_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_97/batch_normalization_908/batchnorm/mulMul9sequential_97/batch_normalization_908/batchnorm/Rsqrt:y:0Jsequential_97/batch_normalization_908/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ò
5sequential_97/batch_normalization_908/batchnorm/mul_1Mul)sequential_97/dense_1005/BiasAdd:output:07sequential_97/batch_normalization_908/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_97/batch_normalization_908/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_97_batch_normalization_908_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_97/batch_normalization_908/batchnorm/mul_2MulHsequential_97/batch_normalization_908/batchnorm/ReadVariableOp_1:value:07sequential_97/batch_normalization_908/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_97/batch_normalization_908/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_97_batch_normalization_908_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_97/batch_normalization_908/batchnorm/subSubHsequential_97/batch_normalization_908/batchnorm/ReadVariableOp_2:value:09sequential_97/batch_normalization_908/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_97/batch_normalization_908/batchnorm/add_1AddV29sequential_97/batch_normalization_908/batchnorm/mul_1:z:07sequential_97/batch_normalization_908/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_97/leaky_re_lu_908/LeakyRelu	LeakyRelu9sequential_97/batch_normalization_908/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¦
.sequential_97/dense_1006/MatMul/ReadVariableOpReadVariableOp7sequential_97_dense_1006_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
sequential_97/dense_1006/MatMulMatMul5sequential_97/leaky_re_lu_908/LeakyRelu:activations:06sequential_97/dense_1006/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/sequential_97/dense_1006/BiasAdd/ReadVariableOpReadVariableOp8sequential_97_dense_1006_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 sequential_97/dense_1006/BiasAddBiasAdd)sequential_97/dense_1006/MatMul:product:07sequential_97/dense_1006/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_97/batch_normalization_909/batchnorm/ReadVariableOpReadVariableOpGsequential_97_batch_normalization_909_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_97/batch_normalization_909/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_97/batch_normalization_909/batchnorm/addAddV2Fsequential_97/batch_normalization_909/batchnorm/ReadVariableOp:value:0>sequential_97/batch_normalization_909/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_97/batch_normalization_909/batchnorm/RsqrtRsqrt7sequential_97/batch_normalization_909/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_97/batch_normalization_909/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_97_batch_normalization_909_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_97/batch_normalization_909/batchnorm/mulMul9sequential_97/batch_normalization_909/batchnorm/Rsqrt:y:0Jsequential_97/batch_normalization_909/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ò
5sequential_97/batch_normalization_909/batchnorm/mul_1Mul)sequential_97/dense_1006/BiasAdd:output:07sequential_97/batch_normalization_909/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_97/batch_normalization_909/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_97_batch_normalization_909_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_97/batch_normalization_909/batchnorm/mul_2MulHsequential_97/batch_normalization_909/batchnorm/ReadVariableOp_1:value:07sequential_97/batch_normalization_909/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_97/batch_normalization_909/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_97_batch_normalization_909_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_97/batch_normalization_909/batchnorm/subSubHsequential_97/batch_normalization_909/batchnorm/ReadVariableOp_2:value:09sequential_97/batch_normalization_909/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_97/batch_normalization_909/batchnorm/add_1AddV29sequential_97/batch_normalization_909/batchnorm/mul_1:z:07sequential_97/batch_normalization_909/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_97/leaky_re_lu_909/LeakyRelu	LeakyRelu9sequential_97/batch_normalization_909/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¦
.sequential_97/dense_1007/MatMul/ReadVariableOpReadVariableOp7sequential_97_dense_1007_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
sequential_97/dense_1007/MatMulMatMul5sequential_97/leaky_re_lu_909/LeakyRelu:activations:06sequential_97/dense_1007/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/sequential_97/dense_1007/BiasAdd/ReadVariableOpReadVariableOp8sequential_97_dense_1007_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 sequential_97/dense_1007/BiasAddBiasAdd)sequential_97/dense_1007/MatMul:product:07sequential_97/dense_1007/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_97/batch_normalization_910/batchnorm/ReadVariableOpReadVariableOpGsequential_97_batch_normalization_910_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_97/batch_normalization_910/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_97/batch_normalization_910/batchnorm/addAddV2Fsequential_97/batch_normalization_910/batchnorm/ReadVariableOp:value:0>sequential_97/batch_normalization_910/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_97/batch_normalization_910/batchnorm/RsqrtRsqrt7sequential_97/batch_normalization_910/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_97/batch_normalization_910/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_97_batch_normalization_910_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_97/batch_normalization_910/batchnorm/mulMul9sequential_97/batch_normalization_910/batchnorm/Rsqrt:y:0Jsequential_97/batch_normalization_910/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ò
5sequential_97/batch_normalization_910/batchnorm/mul_1Mul)sequential_97/dense_1007/BiasAdd:output:07sequential_97/batch_normalization_910/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_97/batch_normalization_910/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_97_batch_normalization_910_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_97/batch_normalization_910/batchnorm/mul_2MulHsequential_97/batch_normalization_910/batchnorm/ReadVariableOp_1:value:07sequential_97/batch_normalization_910/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_97/batch_normalization_910/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_97_batch_normalization_910_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_97/batch_normalization_910/batchnorm/subSubHsequential_97/batch_normalization_910/batchnorm/ReadVariableOp_2:value:09sequential_97/batch_normalization_910/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_97/batch_normalization_910/batchnorm/add_1AddV29sequential_97/batch_normalization_910/batchnorm/mul_1:z:07sequential_97/batch_normalization_910/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_97/leaky_re_lu_910/LeakyRelu	LeakyRelu9sequential_97/batch_normalization_910/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¦
.sequential_97/dense_1008/MatMul/ReadVariableOpReadVariableOp7sequential_97_dense_1008_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
sequential_97/dense_1008/MatMulMatMul5sequential_97/leaky_re_lu_910/LeakyRelu:activations:06sequential_97/dense_1008/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/sequential_97/dense_1008/BiasAdd/ReadVariableOpReadVariableOp8sequential_97_dense_1008_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 sequential_97/dense_1008/BiasAddBiasAdd)sequential_97/dense_1008/MatMul:product:07sequential_97/dense_1008/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_97/batch_normalization_911/batchnorm/ReadVariableOpReadVariableOpGsequential_97_batch_normalization_911_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_97/batch_normalization_911/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_97/batch_normalization_911/batchnorm/addAddV2Fsequential_97/batch_normalization_911/batchnorm/ReadVariableOp:value:0>sequential_97/batch_normalization_911/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_97/batch_normalization_911/batchnorm/RsqrtRsqrt7sequential_97/batch_normalization_911/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_97/batch_normalization_911/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_97_batch_normalization_911_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_97/batch_normalization_911/batchnorm/mulMul9sequential_97/batch_normalization_911/batchnorm/Rsqrt:y:0Jsequential_97/batch_normalization_911/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ò
5sequential_97/batch_normalization_911/batchnorm/mul_1Mul)sequential_97/dense_1008/BiasAdd:output:07sequential_97/batch_normalization_911/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_97/batch_normalization_911/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_97_batch_normalization_911_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_97/batch_normalization_911/batchnorm/mul_2MulHsequential_97/batch_normalization_911/batchnorm/ReadVariableOp_1:value:07sequential_97/batch_normalization_911/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_97/batch_normalization_911/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_97_batch_normalization_911_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_97/batch_normalization_911/batchnorm/subSubHsequential_97/batch_normalization_911/batchnorm/ReadVariableOp_2:value:09sequential_97/batch_normalization_911/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_97/batch_normalization_911/batchnorm/add_1AddV29sequential_97/batch_normalization_911/batchnorm/mul_1:z:07sequential_97/batch_normalization_911/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_97/leaky_re_lu_911/LeakyRelu	LeakyRelu9sequential_97/batch_normalization_911/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¦
.sequential_97/dense_1009/MatMul/ReadVariableOpReadVariableOp7sequential_97_dense_1009_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
sequential_97/dense_1009/MatMulMatMul5sequential_97/leaky_re_lu_911/LeakyRelu:activations:06sequential_97/dense_1009/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/sequential_97/dense_1009/BiasAdd/ReadVariableOpReadVariableOp8sequential_97_dense_1009_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 sequential_97/dense_1009/BiasAddBiasAdd)sequential_97/dense_1009/MatMul:product:07sequential_97/dense_1009/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_97/batch_normalization_912/batchnorm/ReadVariableOpReadVariableOpGsequential_97_batch_normalization_912_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_97/batch_normalization_912/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_97/batch_normalization_912/batchnorm/addAddV2Fsequential_97/batch_normalization_912/batchnorm/ReadVariableOp:value:0>sequential_97/batch_normalization_912/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_97/batch_normalization_912/batchnorm/RsqrtRsqrt7sequential_97/batch_normalization_912/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_97/batch_normalization_912/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_97_batch_normalization_912_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_97/batch_normalization_912/batchnorm/mulMul9sequential_97/batch_normalization_912/batchnorm/Rsqrt:y:0Jsequential_97/batch_normalization_912/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ò
5sequential_97/batch_normalization_912/batchnorm/mul_1Mul)sequential_97/dense_1009/BiasAdd:output:07sequential_97/batch_normalization_912/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_97/batch_normalization_912/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_97_batch_normalization_912_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_97/batch_normalization_912/batchnorm/mul_2MulHsequential_97/batch_normalization_912/batchnorm/ReadVariableOp_1:value:07sequential_97/batch_normalization_912/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_97/batch_normalization_912/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_97_batch_normalization_912_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_97/batch_normalization_912/batchnorm/subSubHsequential_97/batch_normalization_912/batchnorm/ReadVariableOp_2:value:09sequential_97/batch_normalization_912/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_97/batch_normalization_912/batchnorm/add_1AddV29sequential_97/batch_normalization_912/batchnorm/mul_1:z:07sequential_97/batch_normalization_912/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_97/leaky_re_lu_912/LeakyRelu	LeakyRelu9sequential_97/batch_normalization_912/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¦
.sequential_97/dense_1010/MatMul/ReadVariableOpReadVariableOp7sequential_97_dense_1010_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
sequential_97/dense_1010/MatMulMatMul5sequential_97/leaky_re_lu_912/LeakyRelu:activations:06sequential_97/dense_1010/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/sequential_97/dense_1010/BiasAdd/ReadVariableOpReadVariableOp8sequential_97_dense_1010_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 sequential_97/dense_1010/BiasAddBiasAdd)sequential_97/dense_1010/MatMul:product:07sequential_97/dense_1010/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_97/batch_normalization_913/batchnorm/ReadVariableOpReadVariableOpGsequential_97_batch_normalization_913_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_97/batch_normalization_913/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_97/batch_normalization_913/batchnorm/addAddV2Fsequential_97/batch_normalization_913/batchnorm/ReadVariableOp:value:0>sequential_97/batch_normalization_913/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_97/batch_normalization_913/batchnorm/RsqrtRsqrt7sequential_97/batch_normalization_913/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_97/batch_normalization_913/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_97_batch_normalization_913_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_97/batch_normalization_913/batchnorm/mulMul9sequential_97/batch_normalization_913/batchnorm/Rsqrt:y:0Jsequential_97/batch_normalization_913/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ò
5sequential_97/batch_normalization_913/batchnorm/mul_1Mul)sequential_97/dense_1010/BiasAdd:output:07sequential_97/batch_normalization_913/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_97/batch_normalization_913/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_97_batch_normalization_913_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_97/batch_normalization_913/batchnorm/mul_2MulHsequential_97/batch_normalization_913/batchnorm/ReadVariableOp_1:value:07sequential_97/batch_normalization_913/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_97/batch_normalization_913/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_97_batch_normalization_913_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_97/batch_normalization_913/batchnorm/subSubHsequential_97/batch_normalization_913/batchnorm/ReadVariableOp_2:value:09sequential_97/batch_normalization_913/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_97/batch_normalization_913/batchnorm/add_1AddV29sequential_97/batch_normalization_913/batchnorm/mul_1:z:07sequential_97/batch_normalization_913/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_97/leaky_re_lu_913/LeakyRelu	LeakyRelu9sequential_97/batch_normalization_913/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¦
.sequential_97/dense_1011/MatMul/ReadVariableOpReadVariableOp7sequential_97_dense_1011_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ê
sequential_97/dense_1011/MatMulMatMul5sequential_97/leaky_re_lu_913/LeakyRelu:activations:06sequential_97/dense_1011/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/sequential_97/dense_1011/BiasAdd/ReadVariableOpReadVariableOp8sequential_97_dense_1011_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 sequential_97/dense_1011/BiasAddBiasAdd)sequential_97/dense_1011/MatMul:product:07sequential_97/dense_1011/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
IdentityIdentity)sequential_97/dense_1011/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp?^sequential_97/batch_normalization_904/batchnorm/ReadVariableOpA^sequential_97/batch_normalization_904/batchnorm/ReadVariableOp_1A^sequential_97/batch_normalization_904/batchnorm/ReadVariableOp_2C^sequential_97/batch_normalization_904/batchnorm/mul/ReadVariableOp?^sequential_97/batch_normalization_905/batchnorm/ReadVariableOpA^sequential_97/batch_normalization_905/batchnorm/ReadVariableOp_1A^sequential_97/batch_normalization_905/batchnorm/ReadVariableOp_2C^sequential_97/batch_normalization_905/batchnorm/mul/ReadVariableOp?^sequential_97/batch_normalization_906/batchnorm/ReadVariableOpA^sequential_97/batch_normalization_906/batchnorm/ReadVariableOp_1A^sequential_97/batch_normalization_906/batchnorm/ReadVariableOp_2C^sequential_97/batch_normalization_906/batchnorm/mul/ReadVariableOp?^sequential_97/batch_normalization_907/batchnorm/ReadVariableOpA^sequential_97/batch_normalization_907/batchnorm/ReadVariableOp_1A^sequential_97/batch_normalization_907/batchnorm/ReadVariableOp_2C^sequential_97/batch_normalization_907/batchnorm/mul/ReadVariableOp?^sequential_97/batch_normalization_908/batchnorm/ReadVariableOpA^sequential_97/batch_normalization_908/batchnorm/ReadVariableOp_1A^sequential_97/batch_normalization_908/batchnorm/ReadVariableOp_2C^sequential_97/batch_normalization_908/batchnorm/mul/ReadVariableOp?^sequential_97/batch_normalization_909/batchnorm/ReadVariableOpA^sequential_97/batch_normalization_909/batchnorm/ReadVariableOp_1A^sequential_97/batch_normalization_909/batchnorm/ReadVariableOp_2C^sequential_97/batch_normalization_909/batchnorm/mul/ReadVariableOp?^sequential_97/batch_normalization_910/batchnorm/ReadVariableOpA^sequential_97/batch_normalization_910/batchnorm/ReadVariableOp_1A^sequential_97/batch_normalization_910/batchnorm/ReadVariableOp_2C^sequential_97/batch_normalization_910/batchnorm/mul/ReadVariableOp?^sequential_97/batch_normalization_911/batchnorm/ReadVariableOpA^sequential_97/batch_normalization_911/batchnorm/ReadVariableOp_1A^sequential_97/batch_normalization_911/batchnorm/ReadVariableOp_2C^sequential_97/batch_normalization_911/batchnorm/mul/ReadVariableOp?^sequential_97/batch_normalization_912/batchnorm/ReadVariableOpA^sequential_97/batch_normalization_912/batchnorm/ReadVariableOp_1A^sequential_97/batch_normalization_912/batchnorm/ReadVariableOp_2C^sequential_97/batch_normalization_912/batchnorm/mul/ReadVariableOp?^sequential_97/batch_normalization_913/batchnorm/ReadVariableOpA^sequential_97/batch_normalization_913/batchnorm/ReadVariableOp_1A^sequential_97/batch_normalization_913/batchnorm/ReadVariableOp_2C^sequential_97/batch_normalization_913/batchnorm/mul/ReadVariableOp0^sequential_97/dense_1001/BiasAdd/ReadVariableOp/^sequential_97/dense_1001/MatMul/ReadVariableOp0^sequential_97/dense_1002/BiasAdd/ReadVariableOp/^sequential_97/dense_1002/MatMul/ReadVariableOp0^sequential_97/dense_1003/BiasAdd/ReadVariableOp/^sequential_97/dense_1003/MatMul/ReadVariableOp0^sequential_97/dense_1004/BiasAdd/ReadVariableOp/^sequential_97/dense_1004/MatMul/ReadVariableOp0^sequential_97/dense_1005/BiasAdd/ReadVariableOp/^sequential_97/dense_1005/MatMul/ReadVariableOp0^sequential_97/dense_1006/BiasAdd/ReadVariableOp/^sequential_97/dense_1006/MatMul/ReadVariableOp0^sequential_97/dense_1007/BiasAdd/ReadVariableOp/^sequential_97/dense_1007/MatMul/ReadVariableOp0^sequential_97/dense_1008/BiasAdd/ReadVariableOp/^sequential_97/dense_1008/MatMul/ReadVariableOp0^sequential_97/dense_1009/BiasAdd/ReadVariableOp/^sequential_97/dense_1009/MatMul/ReadVariableOp0^sequential_97/dense_1010/BiasAdd/ReadVariableOp/^sequential_97/dense_1010/MatMul/ReadVariableOp0^sequential_97/dense_1011/BiasAdd/ReadVariableOp/^sequential_97/dense_1011/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¸
_input_shapes¦
£:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
>sequential_97/batch_normalization_904/batchnorm/ReadVariableOp>sequential_97/batch_normalization_904/batchnorm/ReadVariableOp2
@sequential_97/batch_normalization_904/batchnorm/ReadVariableOp_1@sequential_97/batch_normalization_904/batchnorm/ReadVariableOp_12
@sequential_97/batch_normalization_904/batchnorm/ReadVariableOp_2@sequential_97/batch_normalization_904/batchnorm/ReadVariableOp_22
Bsequential_97/batch_normalization_904/batchnorm/mul/ReadVariableOpBsequential_97/batch_normalization_904/batchnorm/mul/ReadVariableOp2
>sequential_97/batch_normalization_905/batchnorm/ReadVariableOp>sequential_97/batch_normalization_905/batchnorm/ReadVariableOp2
@sequential_97/batch_normalization_905/batchnorm/ReadVariableOp_1@sequential_97/batch_normalization_905/batchnorm/ReadVariableOp_12
@sequential_97/batch_normalization_905/batchnorm/ReadVariableOp_2@sequential_97/batch_normalization_905/batchnorm/ReadVariableOp_22
Bsequential_97/batch_normalization_905/batchnorm/mul/ReadVariableOpBsequential_97/batch_normalization_905/batchnorm/mul/ReadVariableOp2
>sequential_97/batch_normalization_906/batchnorm/ReadVariableOp>sequential_97/batch_normalization_906/batchnorm/ReadVariableOp2
@sequential_97/batch_normalization_906/batchnorm/ReadVariableOp_1@sequential_97/batch_normalization_906/batchnorm/ReadVariableOp_12
@sequential_97/batch_normalization_906/batchnorm/ReadVariableOp_2@sequential_97/batch_normalization_906/batchnorm/ReadVariableOp_22
Bsequential_97/batch_normalization_906/batchnorm/mul/ReadVariableOpBsequential_97/batch_normalization_906/batchnorm/mul/ReadVariableOp2
>sequential_97/batch_normalization_907/batchnorm/ReadVariableOp>sequential_97/batch_normalization_907/batchnorm/ReadVariableOp2
@sequential_97/batch_normalization_907/batchnorm/ReadVariableOp_1@sequential_97/batch_normalization_907/batchnorm/ReadVariableOp_12
@sequential_97/batch_normalization_907/batchnorm/ReadVariableOp_2@sequential_97/batch_normalization_907/batchnorm/ReadVariableOp_22
Bsequential_97/batch_normalization_907/batchnorm/mul/ReadVariableOpBsequential_97/batch_normalization_907/batchnorm/mul/ReadVariableOp2
>sequential_97/batch_normalization_908/batchnorm/ReadVariableOp>sequential_97/batch_normalization_908/batchnorm/ReadVariableOp2
@sequential_97/batch_normalization_908/batchnorm/ReadVariableOp_1@sequential_97/batch_normalization_908/batchnorm/ReadVariableOp_12
@sequential_97/batch_normalization_908/batchnorm/ReadVariableOp_2@sequential_97/batch_normalization_908/batchnorm/ReadVariableOp_22
Bsequential_97/batch_normalization_908/batchnorm/mul/ReadVariableOpBsequential_97/batch_normalization_908/batchnorm/mul/ReadVariableOp2
>sequential_97/batch_normalization_909/batchnorm/ReadVariableOp>sequential_97/batch_normalization_909/batchnorm/ReadVariableOp2
@sequential_97/batch_normalization_909/batchnorm/ReadVariableOp_1@sequential_97/batch_normalization_909/batchnorm/ReadVariableOp_12
@sequential_97/batch_normalization_909/batchnorm/ReadVariableOp_2@sequential_97/batch_normalization_909/batchnorm/ReadVariableOp_22
Bsequential_97/batch_normalization_909/batchnorm/mul/ReadVariableOpBsequential_97/batch_normalization_909/batchnorm/mul/ReadVariableOp2
>sequential_97/batch_normalization_910/batchnorm/ReadVariableOp>sequential_97/batch_normalization_910/batchnorm/ReadVariableOp2
@sequential_97/batch_normalization_910/batchnorm/ReadVariableOp_1@sequential_97/batch_normalization_910/batchnorm/ReadVariableOp_12
@sequential_97/batch_normalization_910/batchnorm/ReadVariableOp_2@sequential_97/batch_normalization_910/batchnorm/ReadVariableOp_22
Bsequential_97/batch_normalization_910/batchnorm/mul/ReadVariableOpBsequential_97/batch_normalization_910/batchnorm/mul/ReadVariableOp2
>sequential_97/batch_normalization_911/batchnorm/ReadVariableOp>sequential_97/batch_normalization_911/batchnorm/ReadVariableOp2
@sequential_97/batch_normalization_911/batchnorm/ReadVariableOp_1@sequential_97/batch_normalization_911/batchnorm/ReadVariableOp_12
@sequential_97/batch_normalization_911/batchnorm/ReadVariableOp_2@sequential_97/batch_normalization_911/batchnorm/ReadVariableOp_22
Bsequential_97/batch_normalization_911/batchnorm/mul/ReadVariableOpBsequential_97/batch_normalization_911/batchnorm/mul/ReadVariableOp2
>sequential_97/batch_normalization_912/batchnorm/ReadVariableOp>sequential_97/batch_normalization_912/batchnorm/ReadVariableOp2
@sequential_97/batch_normalization_912/batchnorm/ReadVariableOp_1@sequential_97/batch_normalization_912/batchnorm/ReadVariableOp_12
@sequential_97/batch_normalization_912/batchnorm/ReadVariableOp_2@sequential_97/batch_normalization_912/batchnorm/ReadVariableOp_22
Bsequential_97/batch_normalization_912/batchnorm/mul/ReadVariableOpBsequential_97/batch_normalization_912/batchnorm/mul/ReadVariableOp2
>sequential_97/batch_normalization_913/batchnorm/ReadVariableOp>sequential_97/batch_normalization_913/batchnorm/ReadVariableOp2
@sequential_97/batch_normalization_913/batchnorm/ReadVariableOp_1@sequential_97/batch_normalization_913/batchnorm/ReadVariableOp_12
@sequential_97/batch_normalization_913/batchnorm/ReadVariableOp_2@sequential_97/batch_normalization_913/batchnorm/ReadVariableOp_22
Bsequential_97/batch_normalization_913/batchnorm/mul/ReadVariableOpBsequential_97/batch_normalization_913/batchnorm/mul/ReadVariableOp2b
/sequential_97/dense_1001/BiasAdd/ReadVariableOp/sequential_97/dense_1001/BiasAdd/ReadVariableOp2`
.sequential_97/dense_1001/MatMul/ReadVariableOp.sequential_97/dense_1001/MatMul/ReadVariableOp2b
/sequential_97/dense_1002/BiasAdd/ReadVariableOp/sequential_97/dense_1002/BiasAdd/ReadVariableOp2`
.sequential_97/dense_1002/MatMul/ReadVariableOp.sequential_97/dense_1002/MatMul/ReadVariableOp2b
/sequential_97/dense_1003/BiasAdd/ReadVariableOp/sequential_97/dense_1003/BiasAdd/ReadVariableOp2`
.sequential_97/dense_1003/MatMul/ReadVariableOp.sequential_97/dense_1003/MatMul/ReadVariableOp2b
/sequential_97/dense_1004/BiasAdd/ReadVariableOp/sequential_97/dense_1004/BiasAdd/ReadVariableOp2`
.sequential_97/dense_1004/MatMul/ReadVariableOp.sequential_97/dense_1004/MatMul/ReadVariableOp2b
/sequential_97/dense_1005/BiasAdd/ReadVariableOp/sequential_97/dense_1005/BiasAdd/ReadVariableOp2`
.sequential_97/dense_1005/MatMul/ReadVariableOp.sequential_97/dense_1005/MatMul/ReadVariableOp2b
/sequential_97/dense_1006/BiasAdd/ReadVariableOp/sequential_97/dense_1006/BiasAdd/ReadVariableOp2`
.sequential_97/dense_1006/MatMul/ReadVariableOp.sequential_97/dense_1006/MatMul/ReadVariableOp2b
/sequential_97/dense_1007/BiasAdd/ReadVariableOp/sequential_97/dense_1007/BiasAdd/ReadVariableOp2`
.sequential_97/dense_1007/MatMul/ReadVariableOp.sequential_97/dense_1007/MatMul/ReadVariableOp2b
/sequential_97/dense_1008/BiasAdd/ReadVariableOp/sequential_97/dense_1008/BiasAdd/ReadVariableOp2`
.sequential_97/dense_1008/MatMul/ReadVariableOp.sequential_97/dense_1008/MatMul/ReadVariableOp2b
/sequential_97/dense_1009/BiasAdd/ReadVariableOp/sequential_97/dense_1009/BiasAdd/ReadVariableOp2`
.sequential_97/dense_1009/MatMul/ReadVariableOp.sequential_97/dense_1009/MatMul/ReadVariableOp2b
/sequential_97/dense_1010/BiasAdd/ReadVariableOp/sequential_97/dense_1010/BiasAdd/ReadVariableOp2`
.sequential_97/dense_1010/MatMul/ReadVariableOp.sequential_97/dense_1010/MatMul/ReadVariableOp2b
/sequential_97/dense_1011/BiasAdd/ReadVariableOp/sequential_97/dense_1011/BiasAdd/ReadVariableOp2`
.sequential_97/dense_1011/MatMul/ReadVariableOp.sequential_97/dense_1011/MatMul/ReadVariableOp:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_97_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ê	
ø
G__inference_dense_1005_layer_call_and_return_conditional_losses_1003354

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_911_layer_call_and_return_conditional_losses_1003771

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_909_layer_call_and_return_conditional_losses_1003553

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_907_layer_call_and_return_conditional_losses_1003291

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_912_layer_call_and_return_conditional_losses_1000171

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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
S__inference_batch_normalization_909_layer_call_and_return_conditional_losses_999925

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_dense_1001_layer_call_fn_1002908

inputs
unknown:5
	unknown_0:5
identity¢StatefulPartitionedCallÜ
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
GPU 2J 8 *P
fKRI
G__inference_dense_1001_layer_call_and_return_conditional_losses_1000288o
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
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_dense_1009_layer_call_fn_1003780

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1009_layer_call_and_return_conditional_losses_1000544o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1001_layer_call_and_return_conditional_losses_1000288

inputs0
matmul_readvariableop_resource:5-
biasadd_readvariableop_resource:5
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_911_layer_call_fn_1003707

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_911_layer_call_and_return_conditional_losses_1000089o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«§

J__inference_sequential_97_layer_call_and_return_conditional_losses_1001217

inputs
normalization_97_sub_y
normalization_97_sqrt_x$
dense_1001_1001061:5 
dense_1001_1001063:5-
batch_normalization_904_1001066:5-
batch_normalization_904_1001068:5-
batch_normalization_904_1001070:5-
batch_normalization_904_1001072:5$
dense_1002_1001076:55 
dense_1002_1001078:5-
batch_normalization_905_1001081:5-
batch_normalization_905_1001083:5-
batch_normalization_905_1001085:5-
batch_normalization_905_1001087:5$
dense_1003_1001091:5 
dense_1003_1001093:-
batch_normalization_906_1001096:-
batch_normalization_906_1001098:-
batch_normalization_906_1001100:-
batch_normalization_906_1001102:$
dense_1004_1001106: 
dense_1004_1001108:-
batch_normalization_907_1001111:-
batch_normalization_907_1001113:-
batch_normalization_907_1001115:-
batch_normalization_907_1001117:$
dense_1005_1001121: 
dense_1005_1001123:-
batch_normalization_908_1001126:-
batch_normalization_908_1001128:-
batch_normalization_908_1001130:-
batch_normalization_908_1001132:$
dense_1006_1001136: 
dense_1006_1001138:-
batch_normalization_909_1001141:-
batch_normalization_909_1001143:-
batch_normalization_909_1001145:-
batch_normalization_909_1001147:$
dense_1007_1001151: 
dense_1007_1001153:-
batch_normalization_910_1001156:-
batch_normalization_910_1001158:-
batch_normalization_910_1001160:-
batch_normalization_910_1001162:$
dense_1008_1001166: 
dense_1008_1001168:-
batch_normalization_911_1001171:-
batch_normalization_911_1001173:-
batch_normalization_911_1001175:-
batch_normalization_911_1001177:$
dense_1009_1001181: 
dense_1009_1001183:-
batch_normalization_912_1001186:-
batch_normalization_912_1001188:-
batch_normalization_912_1001190:-
batch_normalization_912_1001192:$
dense_1010_1001196: 
dense_1010_1001198:-
batch_normalization_913_1001201:-
batch_normalization_913_1001203:-
batch_normalization_913_1001205:-
batch_normalization_913_1001207:$
dense_1011_1001211: 
dense_1011_1001213:
identity¢/batch_normalization_904/StatefulPartitionedCall¢/batch_normalization_905/StatefulPartitionedCall¢/batch_normalization_906/StatefulPartitionedCall¢/batch_normalization_907/StatefulPartitionedCall¢/batch_normalization_908/StatefulPartitionedCall¢/batch_normalization_909/StatefulPartitionedCall¢/batch_normalization_910/StatefulPartitionedCall¢/batch_normalization_911/StatefulPartitionedCall¢/batch_normalization_912/StatefulPartitionedCall¢/batch_normalization_913/StatefulPartitionedCall¢"dense_1001/StatefulPartitionedCall¢"dense_1002/StatefulPartitionedCall¢"dense_1003/StatefulPartitionedCall¢"dense_1004/StatefulPartitionedCall¢"dense_1005/StatefulPartitionedCall¢"dense_1006/StatefulPartitionedCall¢"dense_1007/StatefulPartitionedCall¢"dense_1008/StatefulPartitionedCall¢"dense_1009/StatefulPartitionedCall¢"dense_1010/StatefulPartitionedCall¢"dense_1011/StatefulPartitionedCallm
normalization_97/subSubinputsnormalization_97_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_97/SqrtSqrtnormalization_97_sqrt_x*
T0*
_output_shapes

:_
normalization_97/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_97/MaximumMaximumnormalization_97/Sqrt:y:0#normalization_97/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_97/truedivRealDivnormalization_97/sub:z:0normalization_97/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"dense_1001/StatefulPartitionedCallStatefulPartitionedCallnormalization_97/truediv:z:0dense_1001_1001061dense_1001_1001063*
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
GPU 2J 8 *P
fKRI
G__inference_dense_1001_layer_call_and_return_conditional_losses_1000288
/batch_normalization_904/StatefulPartitionedCallStatefulPartitionedCall+dense_1001/StatefulPartitionedCall:output:0batch_normalization_904_1001066batch_normalization_904_1001068batch_normalization_904_1001070batch_normalization_904_1001072*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_904_layer_call_and_return_conditional_losses_999515ù
leaky_re_lu_904/PartitionedCallPartitionedCall8batch_normalization_904/StatefulPartitionedCall:output:0*
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
L__inference_leaky_re_lu_904_layer_call_and_return_conditional_losses_1000308
"dense_1002/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_904/PartitionedCall:output:0dense_1002_1001076dense_1002_1001078*
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
GPU 2J 8 *P
fKRI
G__inference_dense_1002_layer_call_and_return_conditional_losses_1000320
/batch_normalization_905/StatefulPartitionedCallStatefulPartitionedCall+dense_1002/StatefulPartitionedCall:output:0batch_normalization_905_1001081batch_normalization_905_1001083batch_normalization_905_1001085batch_normalization_905_1001087*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_905_layer_call_and_return_conditional_losses_999597ù
leaky_re_lu_905/PartitionedCallPartitionedCall8batch_normalization_905/StatefulPartitionedCall:output:0*
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
L__inference_leaky_re_lu_905_layer_call_and_return_conditional_losses_1000340
"dense_1003/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_905/PartitionedCall:output:0dense_1003_1001091dense_1003_1001093*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1003_layer_call_and_return_conditional_losses_1000352
/batch_normalization_906/StatefulPartitionedCallStatefulPartitionedCall+dense_1003/StatefulPartitionedCall:output:0batch_normalization_906_1001096batch_normalization_906_1001098batch_normalization_906_1001100batch_normalization_906_1001102*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_906_layer_call_and_return_conditional_losses_999679ù
leaky_re_lu_906/PartitionedCallPartitionedCall8batch_normalization_906/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_906_layer_call_and_return_conditional_losses_1000372
"dense_1004/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_906/PartitionedCall:output:0dense_1004_1001106dense_1004_1001108*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1004_layer_call_and_return_conditional_losses_1000384
/batch_normalization_907/StatefulPartitionedCallStatefulPartitionedCall+dense_1004/StatefulPartitionedCall:output:0batch_normalization_907_1001111batch_normalization_907_1001113batch_normalization_907_1001115batch_normalization_907_1001117*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_907_layer_call_and_return_conditional_losses_999761ù
leaky_re_lu_907/PartitionedCallPartitionedCall8batch_normalization_907/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_907_layer_call_and_return_conditional_losses_1000404
"dense_1005/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_907/PartitionedCall:output:0dense_1005_1001121dense_1005_1001123*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1005_layer_call_and_return_conditional_losses_1000416
/batch_normalization_908/StatefulPartitionedCallStatefulPartitionedCall+dense_1005/StatefulPartitionedCall:output:0batch_normalization_908_1001126batch_normalization_908_1001128batch_normalization_908_1001130batch_normalization_908_1001132*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_908_layer_call_and_return_conditional_losses_999843ù
leaky_re_lu_908/PartitionedCallPartitionedCall8batch_normalization_908/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_908_layer_call_and_return_conditional_losses_1000436
"dense_1006/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_908/PartitionedCall:output:0dense_1006_1001136dense_1006_1001138*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1006_layer_call_and_return_conditional_losses_1000448
/batch_normalization_909/StatefulPartitionedCallStatefulPartitionedCall+dense_1006/StatefulPartitionedCall:output:0batch_normalization_909_1001141batch_normalization_909_1001143batch_normalization_909_1001145batch_normalization_909_1001147*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_909_layer_call_and_return_conditional_losses_999925ù
leaky_re_lu_909/PartitionedCallPartitionedCall8batch_normalization_909/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_909_layer_call_and_return_conditional_losses_1000468
"dense_1007/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_909/PartitionedCall:output:0dense_1007_1001151dense_1007_1001153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1007_layer_call_and_return_conditional_losses_1000480
/batch_normalization_910/StatefulPartitionedCallStatefulPartitionedCall+dense_1007/StatefulPartitionedCall:output:0batch_normalization_910_1001156batch_normalization_910_1001158batch_normalization_910_1001160batch_normalization_910_1001162*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_910_layer_call_and_return_conditional_losses_1000007ù
leaky_re_lu_910/PartitionedCallPartitionedCall8batch_normalization_910/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_910_layer_call_and_return_conditional_losses_1000500
"dense_1008/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_910/PartitionedCall:output:0dense_1008_1001166dense_1008_1001168*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1008_layer_call_and_return_conditional_losses_1000512
/batch_normalization_911/StatefulPartitionedCallStatefulPartitionedCall+dense_1008/StatefulPartitionedCall:output:0batch_normalization_911_1001171batch_normalization_911_1001173batch_normalization_911_1001175batch_normalization_911_1001177*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_911_layer_call_and_return_conditional_losses_1000089ù
leaky_re_lu_911/PartitionedCallPartitionedCall8batch_normalization_911/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_911_layer_call_and_return_conditional_losses_1000532
"dense_1009/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_911/PartitionedCall:output:0dense_1009_1001181dense_1009_1001183*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1009_layer_call_and_return_conditional_losses_1000544
/batch_normalization_912/StatefulPartitionedCallStatefulPartitionedCall+dense_1009/StatefulPartitionedCall:output:0batch_normalization_912_1001186batch_normalization_912_1001188batch_normalization_912_1001190batch_normalization_912_1001192*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_912_layer_call_and_return_conditional_losses_1000171ù
leaky_re_lu_912/PartitionedCallPartitionedCall8batch_normalization_912/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_912_layer_call_and_return_conditional_losses_1000564
"dense_1010/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_912/PartitionedCall:output:0dense_1010_1001196dense_1010_1001198*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1010_layer_call_and_return_conditional_losses_1000576
/batch_normalization_913/StatefulPartitionedCallStatefulPartitionedCall+dense_1010/StatefulPartitionedCall:output:0batch_normalization_913_1001201batch_normalization_913_1001203batch_normalization_913_1001205batch_normalization_913_1001207*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_913_layer_call_and_return_conditional_losses_1000253ù
leaky_re_lu_913/PartitionedCallPartitionedCall8batch_normalization_913/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_913_layer_call_and_return_conditional_losses_1000596
"dense_1011/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_913/PartitionedCall:output:0dense_1011_1001211dense_1011_1001213*
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
GPU 2J 8 *P
fKRI
G__inference_dense_1011_layer_call_and_return_conditional_losses_1000608z
IdentityIdentity+dense_1011/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp0^batch_normalization_904/StatefulPartitionedCall0^batch_normalization_905/StatefulPartitionedCall0^batch_normalization_906/StatefulPartitionedCall0^batch_normalization_907/StatefulPartitionedCall0^batch_normalization_908/StatefulPartitionedCall0^batch_normalization_909/StatefulPartitionedCall0^batch_normalization_910/StatefulPartitionedCall0^batch_normalization_911/StatefulPartitionedCall0^batch_normalization_912/StatefulPartitionedCall0^batch_normalization_913/StatefulPartitionedCall#^dense_1001/StatefulPartitionedCall#^dense_1002/StatefulPartitionedCall#^dense_1003/StatefulPartitionedCall#^dense_1004/StatefulPartitionedCall#^dense_1005/StatefulPartitionedCall#^dense_1006/StatefulPartitionedCall#^dense_1007/StatefulPartitionedCall#^dense_1008/StatefulPartitionedCall#^dense_1009/StatefulPartitionedCall#^dense_1010/StatefulPartitionedCall#^dense_1011/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¸
_input_shapes¦
£:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_904/StatefulPartitionedCall/batch_normalization_904/StatefulPartitionedCall2b
/batch_normalization_905/StatefulPartitionedCall/batch_normalization_905/StatefulPartitionedCall2b
/batch_normalization_906/StatefulPartitionedCall/batch_normalization_906/StatefulPartitionedCall2b
/batch_normalization_907/StatefulPartitionedCall/batch_normalization_907/StatefulPartitionedCall2b
/batch_normalization_908/StatefulPartitionedCall/batch_normalization_908/StatefulPartitionedCall2b
/batch_normalization_909/StatefulPartitionedCall/batch_normalization_909/StatefulPartitionedCall2b
/batch_normalization_910/StatefulPartitionedCall/batch_normalization_910/StatefulPartitionedCall2b
/batch_normalization_911/StatefulPartitionedCall/batch_normalization_911/StatefulPartitionedCall2b
/batch_normalization_912/StatefulPartitionedCall/batch_normalization_912/StatefulPartitionedCall2b
/batch_normalization_913/StatefulPartitionedCall/batch_normalization_913/StatefulPartitionedCall2H
"dense_1001/StatefulPartitionedCall"dense_1001/StatefulPartitionedCall2H
"dense_1002/StatefulPartitionedCall"dense_1002/StatefulPartitionedCall2H
"dense_1003/StatefulPartitionedCall"dense_1003/StatefulPartitionedCall2H
"dense_1004/StatefulPartitionedCall"dense_1004/StatefulPartitionedCall2H
"dense_1005/StatefulPartitionedCall"dense_1005/StatefulPartitionedCall2H
"dense_1006/StatefulPartitionedCall"dense_1006/StatefulPartitionedCall2H
"dense_1007/StatefulPartitionedCall"dense_1007/StatefulPartitionedCall2H
"dense_1008/StatefulPartitionedCall"dense_1008/StatefulPartitionedCall2H
"dense_1009/StatefulPartitionedCall"dense_1009/StatefulPartitionedCall2H
"dense_1010/StatefulPartitionedCall"dense_1010/StatefulPartitionedCall2H
"dense_1011/StatefulPartitionedCall"dense_1011/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ê	
ø
G__inference_dense_1009_layer_call_and_return_conditional_losses_1003790

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_dense_1008_layer_call_fn_1003671

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1008_layer_call_and_return_conditional_losses_1000512o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_910_layer_call_and_return_conditional_losses_999960

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_907_layer_call_and_return_conditional_losses_1003325

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û§
­
J__inference_sequential_97_layer_call_and_return_conditional_losses_1001813
normalization_97_input
normalization_97_sub_y
normalization_97_sqrt_x$
dense_1001_1001657:5 
dense_1001_1001659:5-
batch_normalization_904_1001662:5-
batch_normalization_904_1001664:5-
batch_normalization_904_1001666:5-
batch_normalization_904_1001668:5$
dense_1002_1001672:55 
dense_1002_1001674:5-
batch_normalization_905_1001677:5-
batch_normalization_905_1001679:5-
batch_normalization_905_1001681:5-
batch_normalization_905_1001683:5$
dense_1003_1001687:5 
dense_1003_1001689:-
batch_normalization_906_1001692:-
batch_normalization_906_1001694:-
batch_normalization_906_1001696:-
batch_normalization_906_1001698:$
dense_1004_1001702: 
dense_1004_1001704:-
batch_normalization_907_1001707:-
batch_normalization_907_1001709:-
batch_normalization_907_1001711:-
batch_normalization_907_1001713:$
dense_1005_1001717: 
dense_1005_1001719:-
batch_normalization_908_1001722:-
batch_normalization_908_1001724:-
batch_normalization_908_1001726:-
batch_normalization_908_1001728:$
dense_1006_1001732: 
dense_1006_1001734:-
batch_normalization_909_1001737:-
batch_normalization_909_1001739:-
batch_normalization_909_1001741:-
batch_normalization_909_1001743:$
dense_1007_1001747: 
dense_1007_1001749:-
batch_normalization_910_1001752:-
batch_normalization_910_1001754:-
batch_normalization_910_1001756:-
batch_normalization_910_1001758:$
dense_1008_1001762: 
dense_1008_1001764:-
batch_normalization_911_1001767:-
batch_normalization_911_1001769:-
batch_normalization_911_1001771:-
batch_normalization_911_1001773:$
dense_1009_1001777: 
dense_1009_1001779:-
batch_normalization_912_1001782:-
batch_normalization_912_1001784:-
batch_normalization_912_1001786:-
batch_normalization_912_1001788:$
dense_1010_1001792: 
dense_1010_1001794:-
batch_normalization_913_1001797:-
batch_normalization_913_1001799:-
batch_normalization_913_1001801:-
batch_normalization_913_1001803:$
dense_1011_1001807: 
dense_1011_1001809:
identity¢/batch_normalization_904/StatefulPartitionedCall¢/batch_normalization_905/StatefulPartitionedCall¢/batch_normalization_906/StatefulPartitionedCall¢/batch_normalization_907/StatefulPartitionedCall¢/batch_normalization_908/StatefulPartitionedCall¢/batch_normalization_909/StatefulPartitionedCall¢/batch_normalization_910/StatefulPartitionedCall¢/batch_normalization_911/StatefulPartitionedCall¢/batch_normalization_912/StatefulPartitionedCall¢/batch_normalization_913/StatefulPartitionedCall¢"dense_1001/StatefulPartitionedCall¢"dense_1002/StatefulPartitionedCall¢"dense_1003/StatefulPartitionedCall¢"dense_1004/StatefulPartitionedCall¢"dense_1005/StatefulPartitionedCall¢"dense_1006/StatefulPartitionedCall¢"dense_1007/StatefulPartitionedCall¢"dense_1008/StatefulPartitionedCall¢"dense_1009/StatefulPartitionedCall¢"dense_1010/StatefulPartitionedCall¢"dense_1011/StatefulPartitionedCall}
normalization_97/subSubnormalization_97_inputnormalization_97_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_97/SqrtSqrtnormalization_97_sqrt_x*
T0*
_output_shapes

:_
normalization_97/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_97/MaximumMaximumnormalization_97/Sqrt:y:0#normalization_97/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_97/truedivRealDivnormalization_97/sub:z:0normalization_97/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"dense_1001/StatefulPartitionedCallStatefulPartitionedCallnormalization_97/truediv:z:0dense_1001_1001657dense_1001_1001659*
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
GPU 2J 8 *P
fKRI
G__inference_dense_1001_layer_call_and_return_conditional_losses_1000288
/batch_normalization_904/StatefulPartitionedCallStatefulPartitionedCall+dense_1001/StatefulPartitionedCall:output:0batch_normalization_904_1001662batch_normalization_904_1001664batch_normalization_904_1001666batch_normalization_904_1001668*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_904_layer_call_and_return_conditional_losses_999515ù
leaky_re_lu_904/PartitionedCallPartitionedCall8batch_normalization_904/StatefulPartitionedCall:output:0*
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
L__inference_leaky_re_lu_904_layer_call_and_return_conditional_losses_1000308
"dense_1002/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_904/PartitionedCall:output:0dense_1002_1001672dense_1002_1001674*
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
GPU 2J 8 *P
fKRI
G__inference_dense_1002_layer_call_and_return_conditional_losses_1000320
/batch_normalization_905/StatefulPartitionedCallStatefulPartitionedCall+dense_1002/StatefulPartitionedCall:output:0batch_normalization_905_1001677batch_normalization_905_1001679batch_normalization_905_1001681batch_normalization_905_1001683*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_905_layer_call_and_return_conditional_losses_999597ù
leaky_re_lu_905/PartitionedCallPartitionedCall8batch_normalization_905/StatefulPartitionedCall:output:0*
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
L__inference_leaky_re_lu_905_layer_call_and_return_conditional_losses_1000340
"dense_1003/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_905/PartitionedCall:output:0dense_1003_1001687dense_1003_1001689*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1003_layer_call_and_return_conditional_losses_1000352
/batch_normalization_906/StatefulPartitionedCallStatefulPartitionedCall+dense_1003/StatefulPartitionedCall:output:0batch_normalization_906_1001692batch_normalization_906_1001694batch_normalization_906_1001696batch_normalization_906_1001698*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_906_layer_call_and_return_conditional_losses_999679ù
leaky_re_lu_906/PartitionedCallPartitionedCall8batch_normalization_906/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_906_layer_call_and_return_conditional_losses_1000372
"dense_1004/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_906/PartitionedCall:output:0dense_1004_1001702dense_1004_1001704*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1004_layer_call_and_return_conditional_losses_1000384
/batch_normalization_907/StatefulPartitionedCallStatefulPartitionedCall+dense_1004/StatefulPartitionedCall:output:0batch_normalization_907_1001707batch_normalization_907_1001709batch_normalization_907_1001711batch_normalization_907_1001713*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_907_layer_call_and_return_conditional_losses_999761ù
leaky_re_lu_907/PartitionedCallPartitionedCall8batch_normalization_907/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_907_layer_call_and_return_conditional_losses_1000404
"dense_1005/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_907/PartitionedCall:output:0dense_1005_1001717dense_1005_1001719*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1005_layer_call_and_return_conditional_losses_1000416
/batch_normalization_908/StatefulPartitionedCallStatefulPartitionedCall+dense_1005/StatefulPartitionedCall:output:0batch_normalization_908_1001722batch_normalization_908_1001724batch_normalization_908_1001726batch_normalization_908_1001728*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_908_layer_call_and_return_conditional_losses_999843ù
leaky_re_lu_908/PartitionedCallPartitionedCall8batch_normalization_908/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_908_layer_call_and_return_conditional_losses_1000436
"dense_1006/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_908/PartitionedCall:output:0dense_1006_1001732dense_1006_1001734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1006_layer_call_and_return_conditional_losses_1000448
/batch_normalization_909/StatefulPartitionedCallStatefulPartitionedCall+dense_1006/StatefulPartitionedCall:output:0batch_normalization_909_1001737batch_normalization_909_1001739batch_normalization_909_1001741batch_normalization_909_1001743*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_909_layer_call_and_return_conditional_losses_999925ù
leaky_re_lu_909/PartitionedCallPartitionedCall8batch_normalization_909/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_909_layer_call_and_return_conditional_losses_1000468
"dense_1007/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_909/PartitionedCall:output:0dense_1007_1001747dense_1007_1001749*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1007_layer_call_and_return_conditional_losses_1000480
/batch_normalization_910/StatefulPartitionedCallStatefulPartitionedCall+dense_1007/StatefulPartitionedCall:output:0batch_normalization_910_1001752batch_normalization_910_1001754batch_normalization_910_1001756batch_normalization_910_1001758*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_910_layer_call_and_return_conditional_losses_1000007ù
leaky_re_lu_910/PartitionedCallPartitionedCall8batch_normalization_910/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_910_layer_call_and_return_conditional_losses_1000500
"dense_1008/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_910/PartitionedCall:output:0dense_1008_1001762dense_1008_1001764*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1008_layer_call_and_return_conditional_losses_1000512
/batch_normalization_911/StatefulPartitionedCallStatefulPartitionedCall+dense_1008/StatefulPartitionedCall:output:0batch_normalization_911_1001767batch_normalization_911_1001769batch_normalization_911_1001771batch_normalization_911_1001773*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_911_layer_call_and_return_conditional_losses_1000089ù
leaky_re_lu_911/PartitionedCallPartitionedCall8batch_normalization_911/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_911_layer_call_and_return_conditional_losses_1000532
"dense_1009/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_911/PartitionedCall:output:0dense_1009_1001777dense_1009_1001779*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1009_layer_call_and_return_conditional_losses_1000544
/batch_normalization_912/StatefulPartitionedCallStatefulPartitionedCall+dense_1009/StatefulPartitionedCall:output:0batch_normalization_912_1001782batch_normalization_912_1001784batch_normalization_912_1001786batch_normalization_912_1001788*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_912_layer_call_and_return_conditional_losses_1000171ù
leaky_re_lu_912/PartitionedCallPartitionedCall8batch_normalization_912/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_912_layer_call_and_return_conditional_losses_1000564
"dense_1010/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_912/PartitionedCall:output:0dense_1010_1001792dense_1010_1001794*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1010_layer_call_and_return_conditional_losses_1000576
/batch_normalization_913/StatefulPartitionedCallStatefulPartitionedCall+dense_1010/StatefulPartitionedCall:output:0batch_normalization_913_1001797batch_normalization_913_1001799batch_normalization_913_1001801batch_normalization_913_1001803*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_913_layer_call_and_return_conditional_losses_1000253ù
leaky_re_lu_913/PartitionedCallPartitionedCall8batch_normalization_913/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_913_layer_call_and_return_conditional_losses_1000596
"dense_1011/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_913/PartitionedCall:output:0dense_1011_1001807dense_1011_1001809*
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
GPU 2J 8 *P
fKRI
G__inference_dense_1011_layer_call_and_return_conditional_losses_1000608z
IdentityIdentity+dense_1011/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp0^batch_normalization_904/StatefulPartitionedCall0^batch_normalization_905/StatefulPartitionedCall0^batch_normalization_906/StatefulPartitionedCall0^batch_normalization_907/StatefulPartitionedCall0^batch_normalization_908/StatefulPartitionedCall0^batch_normalization_909/StatefulPartitionedCall0^batch_normalization_910/StatefulPartitionedCall0^batch_normalization_911/StatefulPartitionedCall0^batch_normalization_912/StatefulPartitionedCall0^batch_normalization_913/StatefulPartitionedCall#^dense_1001/StatefulPartitionedCall#^dense_1002/StatefulPartitionedCall#^dense_1003/StatefulPartitionedCall#^dense_1004/StatefulPartitionedCall#^dense_1005/StatefulPartitionedCall#^dense_1006/StatefulPartitionedCall#^dense_1007/StatefulPartitionedCall#^dense_1008/StatefulPartitionedCall#^dense_1009/StatefulPartitionedCall#^dense_1010/StatefulPartitionedCall#^dense_1011/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¸
_input_shapes¦
£:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_904/StatefulPartitionedCall/batch_normalization_904/StatefulPartitionedCall2b
/batch_normalization_905/StatefulPartitionedCall/batch_normalization_905/StatefulPartitionedCall2b
/batch_normalization_906/StatefulPartitionedCall/batch_normalization_906/StatefulPartitionedCall2b
/batch_normalization_907/StatefulPartitionedCall/batch_normalization_907/StatefulPartitionedCall2b
/batch_normalization_908/StatefulPartitionedCall/batch_normalization_908/StatefulPartitionedCall2b
/batch_normalization_909/StatefulPartitionedCall/batch_normalization_909/StatefulPartitionedCall2b
/batch_normalization_910/StatefulPartitionedCall/batch_normalization_910/StatefulPartitionedCall2b
/batch_normalization_911/StatefulPartitionedCall/batch_normalization_911/StatefulPartitionedCall2b
/batch_normalization_912/StatefulPartitionedCall/batch_normalization_912/StatefulPartitionedCall2b
/batch_normalization_913/StatefulPartitionedCall/batch_normalization_913/StatefulPartitionedCall2H
"dense_1001/StatefulPartitionedCall"dense_1001/StatefulPartitionedCall2H
"dense_1002/StatefulPartitionedCall"dense_1002/StatefulPartitionedCall2H
"dense_1003/StatefulPartitionedCall"dense_1003/StatefulPartitionedCall2H
"dense_1004/StatefulPartitionedCall"dense_1004/StatefulPartitionedCall2H
"dense_1005/StatefulPartitionedCall"dense_1005/StatefulPartitionedCall2H
"dense_1006/StatefulPartitionedCall"dense_1006/StatefulPartitionedCall2H
"dense_1007/StatefulPartitionedCall"dense_1007/StatefulPartitionedCall2H
"dense_1008/StatefulPartitionedCall"dense_1008/StatefulPartitionedCall2H
"dense_1009/StatefulPartitionedCall"dense_1009/StatefulPartitionedCall2H
"dense_1010/StatefulPartitionedCall"dense_1010/StatefulPartitionedCall2H
"dense_1011/StatefulPartitionedCall"dense_1011/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_97_input:$ 

_output_shapes

::$ 

_output_shapes

:
æ
h
L__inference_leaky_re_lu_904_layer_call_and_return_conditional_losses_1003008

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
õ

/__inference_sequential_97_layer_call_fn_1001481
normalization_97_input
unknown
	unknown_0
	unknown_1:5
	unknown_2:5
	unknown_3:5
	unknown_4:5
	unknown_5:5
	unknown_6:5
	unknown_7:55
	unknown_8:5
	unknown_9:5

unknown_10:5

unknown_11:5

unknown_12:5

unknown_13:5

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:

unknown_55:

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:

unknown_61:

unknown_62:
identity¢StatefulPartitionedCall´	
StatefulPartitionedCallStatefulPartitionedCallnormalization_97_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*L
_read_only_resource_inputs.
,*	
 !"%&'(+,-.1234789:=>?@*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_97_layer_call_and_return_conditional_losses_1001217o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¸
_input_shapes¦
£:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_97_input:$ 

_output_shapes

::$ 

_output_shapes

:
­
M
1__inference_leaky_re_lu_910_layer_call_fn_1003657

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_910_layer_call_and_return_conditional_losses_1000500`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_dense_1004_layer_call_fn_1003235

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1004_layer_call_and_return_conditional_losses_1000384o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_913_layer_call_and_return_conditional_losses_1000206

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_913_layer_call_and_return_conditional_losses_1000253

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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
«
Ô
9__inference_batch_normalization_904_layer_call_fn_1002944

inputs
unknown:5
	unknown_0:5
	unknown_1:5
	unknown_2:5
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_904_layer_call_and_return_conditional_losses_999515o
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
T__inference_batch_normalization_908_layer_call_and_return_conditional_losses_1003400

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_911_layer_call_and_return_conditional_losses_1000532

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
ù
%__inference_signature_wrapper_1002852
normalization_97_input
unknown
	unknown_0
	unknown_1:5
	unknown_2:5
	unknown_3:5
	unknown_4:5
	unknown_5:5
	unknown_6:5
	unknown_7:55
	unknown_8:5
	unknown_9:5

unknown_10:5

unknown_11:5

unknown_12:5

unknown_13:5

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:

unknown_55:

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:

unknown_61:

unknown_62:
identity¢StatefulPartitionedCall	
StatefulPartitionedCallStatefulPartitionedCallnormalization_97_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>?@*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_999444o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¸
_input_shapes¦
£:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_97_input:$ 

_output_shapes

::$ 

_output_shapes

:
«
Ô
9__inference_batch_normalization_905_layer_call_fn_1003053

inputs
unknown:5
	unknown_0:5
	unknown_1:5
	unknown_2:5
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_905_layer_call_and_return_conditional_losses_999597o
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
%
í
T__inference_batch_normalization_910_layer_call_and_return_conditional_losses_1000007

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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
«
Ô
9__inference_batch_normalization_909_layer_call_fn_1003489

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_909_layer_call_and_return_conditional_losses_999925o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_909_layer_call_fn_1003548

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_909_layer_call_and_return_conditional_losses_1000468`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
Ô
9__inference_batch_normalization_910_layer_call_fn_1003585

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_910_layer_call_and_return_conditional_losses_999960o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_912_layer_call_fn_1003803

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_912_layer_call_and_return_conditional_losses_1000124o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_913_layer_call_and_return_conditional_losses_1003989

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_907_layer_call_and_return_conditional_losses_999714

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_904_layer_call_and_return_conditional_losses_1000308

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
¤ù
ëh
#__inference__traced_restore_1004973
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 6
$assignvariableop_3_dense_1001_kernel:50
"assignvariableop_4_dense_1001_bias:5>
0assignvariableop_5_batch_normalization_904_gamma:5=
/assignvariableop_6_batch_normalization_904_beta:5D
6assignvariableop_7_batch_normalization_904_moving_mean:5H
:assignvariableop_8_batch_normalization_904_moving_variance:56
$assignvariableop_9_dense_1002_kernel:551
#assignvariableop_10_dense_1002_bias:5?
1assignvariableop_11_batch_normalization_905_gamma:5>
0assignvariableop_12_batch_normalization_905_beta:5E
7assignvariableop_13_batch_normalization_905_moving_mean:5I
;assignvariableop_14_batch_normalization_905_moving_variance:57
%assignvariableop_15_dense_1003_kernel:51
#assignvariableop_16_dense_1003_bias:?
1assignvariableop_17_batch_normalization_906_gamma:>
0assignvariableop_18_batch_normalization_906_beta:E
7assignvariableop_19_batch_normalization_906_moving_mean:I
;assignvariableop_20_batch_normalization_906_moving_variance:7
%assignvariableop_21_dense_1004_kernel:1
#assignvariableop_22_dense_1004_bias:?
1assignvariableop_23_batch_normalization_907_gamma:>
0assignvariableop_24_batch_normalization_907_beta:E
7assignvariableop_25_batch_normalization_907_moving_mean:I
;assignvariableop_26_batch_normalization_907_moving_variance:7
%assignvariableop_27_dense_1005_kernel:1
#assignvariableop_28_dense_1005_bias:?
1assignvariableop_29_batch_normalization_908_gamma:>
0assignvariableop_30_batch_normalization_908_beta:E
7assignvariableop_31_batch_normalization_908_moving_mean:I
;assignvariableop_32_batch_normalization_908_moving_variance:7
%assignvariableop_33_dense_1006_kernel:1
#assignvariableop_34_dense_1006_bias:?
1assignvariableop_35_batch_normalization_909_gamma:>
0assignvariableop_36_batch_normalization_909_beta:E
7assignvariableop_37_batch_normalization_909_moving_mean:I
;assignvariableop_38_batch_normalization_909_moving_variance:7
%assignvariableop_39_dense_1007_kernel:1
#assignvariableop_40_dense_1007_bias:?
1assignvariableop_41_batch_normalization_910_gamma:>
0assignvariableop_42_batch_normalization_910_beta:E
7assignvariableop_43_batch_normalization_910_moving_mean:I
;assignvariableop_44_batch_normalization_910_moving_variance:7
%assignvariableop_45_dense_1008_kernel:1
#assignvariableop_46_dense_1008_bias:?
1assignvariableop_47_batch_normalization_911_gamma:>
0assignvariableop_48_batch_normalization_911_beta:E
7assignvariableop_49_batch_normalization_911_moving_mean:I
;assignvariableop_50_batch_normalization_911_moving_variance:7
%assignvariableop_51_dense_1009_kernel:1
#assignvariableop_52_dense_1009_bias:?
1assignvariableop_53_batch_normalization_912_gamma:>
0assignvariableop_54_batch_normalization_912_beta:E
7assignvariableop_55_batch_normalization_912_moving_mean:I
;assignvariableop_56_batch_normalization_912_moving_variance:7
%assignvariableop_57_dense_1010_kernel:1
#assignvariableop_58_dense_1010_bias:?
1assignvariableop_59_batch_normalization_913_gamma:>
0assignvariableop_60_batch_normalization_913_beta:E
7assignvariableop_61_batch_normalization_913_moving_mean:I
;assignvariableop_62_batch_normalization_913_moving_variance:7
%assignvariableop_63_dense_1011_kernel:1
#assignvariableop_64_dense_1011_bias:'
assignvariableop_65_adam_iter:	 )
assignvariableop_66_adam_beta_1: )
assignvariableop_67_adam_beta_2: (
assignvariableop_68_adam_decay: #
assignvariableop_69_total: %
assignvariableop_70_count_1: >
,assignvariableop_71_adam_dense_1001_kernel_m:58
*assignvariableop_72_adam_dense_1001_bias_m:5F
8assignvariableop_73_adam_batch_normalization_904_gamma_m:5E
7assignvariableop_74_adam_batch_normalization_904_beta_m:5>
,assignvariableop_75_adam_dense_1002_kernel_m:558
*assignvariableop_76_adam_dense_1002_bias_m:5F
8assignvariableop_77_adam_batch_normalization_905_gamma_m:5E
7assignvariableop_78_adam_batch_normalization_905_beta_m:5>
,assignvariableop_79_adam_dense_1003_kernel_m:58
*assignvariableop_80_adam_dense_1003_bias_m:F
8assignvariableop_81_adam_batch_normalization_906_gamma_m:E
7assignvariableop_82_adam_batch_normalization_906_beta_m:>
,assignvariableop_83_adam_dense_1004_kernel_m:8
*assignvariableop_84_adam_dense_1004_bias_m:F
8assignvariableop_85_adam_batch_normalization_907_gamma_m:E
7assignvariableop_86_adam_batch_normalization_907_beta_m:>
,assignvariableop_87_adam_dense_1005_kernel_m:8
*assignvariableop_88_adam_dense_1005_bias_m:F
8assignvariableop_89_adam_batch_normalization_908_gamma_m:E
7assignvariableop_90_adam_batch_normalization_908_beta_m:>
,assignvariableop_91_adam_dense_1006_kernel_m:8
*assignvariableop_92_adam_dense_1006_bias_m:F
8assignvariableop_93_adam_batch_normalization_909_gamma_m:E
7assignvariableop_94_adam_batch_normalization_909_beta_m:>
,assignvariableop_95_adam_dense_1007_kernel_m:8
*assignvariableop_96_adam_dense_1007_bias_m:F
8assignvariableop_97_adam_batch_normalization_910_gamma_m:E
7assignvariableop_98_adam_batch_normalization_910_beta_m:>
,assignvariableop_99_adam_dense_1008_kernel_m:9
+assignvariableop_100_adam_dense_1008_bias_m:G
9assignvariableop_101_adam_batch_normalization_911_gamma_m:F
8assignvariableop_102_adam_batch_normalization_911_beta_m:?
-assignvariableop_103_adam_dense_1009_kernel_m:9
+assignvariableop_104_adam_dense_1009_bias_m:G
9assignvariableop_105_adam_batch_normalization_912_gamma_m:F
8assignvariableop_106_adam_batch_normalization_912_beta_m:?
-assignvariableop_107_adam_dense_1010_kernel_m:9
+assignvariableop_108_adam_dense_1010_bias_m:G
9assignvariableop_109_adam_batch_normalization_913_gamma_m:F
8assignvariableop_110_adam_batch_normalization_913_beta_m:?
-assignvariableop_111_adam_dense_1011_kernel_m:9
+assignvariableop_112_adam_dense_1011_bias_m:?
-assignvariableop_113_adam_dense_1001_kernel_v:59
+assignvariableop_114_adam_dense_1001_bias_v:5G
9assignvariableop_115_adam_batch_normalization_904_gamma_v:5F
8assignvariableop_116_adam_batch_normalization_904_beta_v:5?
-assignvariableop_117_adam_dense_1002_kernel_v:559
+assignvariableop_118_adam_dense_1002_bias_v:5G
9assignvariableop_119_adam_batch_normalization_905_gamma_v:5F
8assignvariableop_120_adam_batch_normalization_905_beta_v:5?
-assignvariableop_121_adam_dense_1003_kernel_v:59
+assignvariableop_122_adam_dense_1003_bias_v:G
9assignvariableop_123_adam_batch_normalization_906_gamma_v:F
8assignvariableop_124_adam_batch_normalization_906_beta_v:?
-assignvariableop_125_adam_dense_1004_kernel_v:9
+assignvariableop_126_adam_dense_1004_bias_v:G
9assignvariableop_127_adam_batch_normalization_907_gamma_v:F
8assignvariableop_128_adam_batch_normalization_907_beta_v:?
-assignvariableop_129_adam_dense_1005_kernel_v:9
+assignvariableop_130_adam_dense_1005_bias_v:G
9assignvariableop_131_adam_batch_normalization_908_gamma_v:F
8assignvariableop_132_adam_batch_normalization_908_beta_v:?
-assignvariableop_133_adam_dense_1006_kernel_v:9
+assignvariableop_134_adam_dense_1006_bias_v:G
9assignvariableop_135_adam_batch_normalization_909_gamma_v:F
8assignvariableop_136_adam_batch_normalization_909_beta_v:?
-assignvariableop_137_adam_dense_1007_kernel_v:9
+assignvariableop_138_adam_dense_1007_bias_v:G
9assignvariableop_139_adam_batch_normalization_910_gamma_v:F
8assignvariableop_140_adam_batch_normalization_910_beta_v:?
-assignvariableop_141_adam_dense_1008_kernel_v:9
+assignvariableop_142_adam_dense_1008_bias_v:G
9assignvariableop_143_adam_batch_normalization_911_gamma_v:F
8assignvariableop_144_adam_batch_normalization_911_beta_v:?
-assignvariableop_145_adam_dense_1009_kernel_v:9
+assignvariableop_146_adam_dense_1009_bias_v:G
9assignvariableop_147_adam_batch_normalization_912_gamma_v:F
8assignvariableop_148_adam_batch_normalization_912_beta_v:?
-assignvariableop_149_adam_dense_1010_kernel_v:9
+assignvariableop_150_adam_dense_1010_bias_v:G
9assignvariableop_151_adam_batch_normalization_913_gamma_v:F
8assignvariableop_152_adam_batch_normalization_913_beta_v:?
-assignvariableop_153_adam_dense_1011_kernel_v:9
+assignvariableop_154_adam_dense_1011_bias_v:
identity_156¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_105¢AssignVariableOp_106¢AssignVariableOp_107¢AssignVariableOp_108¢AssignVariableOp_109¢AssignVariableOp_11¢AssignVariableOp_110¢AssignVariableOp_111¢AssignVariableOp_112¢AssignVariableOp_113¢AssignVariableOp_114¢AssignVariableOp_115¢AssignVariableOp_116¢AssignVariableOp_117¢AssignVariableOp_118¢AssignVariableOp_119¢AssignVariableOp_12¢AssignVariableOp_120¢AssignVariableOp_121¢AssignVariableOp_122¢AssignVariableOp_123¢AssignVariableOp_124¢AssignVariableOp_125¢AssignVariableOp_126¢AssignVariableOp_127¢AssignVariableOp_128¢AssignVariableOp_129¢AssignVariableOp_13¢AssignVariableOp_130¢AssignVariableOp_131¢AssignVariableOp_132¢AssignVariableOp_133¢AssignVariableOp_134¢AssignVariableOp_135¢AssignVariableOp_136¢AssignVariableOp_137¢AssignVariableOp_138¢AssignVariableOp_139¢AssignVariableOp_14¢AssignVariableOp_140¢AssignVariableOp_141¢AssignVariableOp_142¢AssignVariableOp_143¢AssignVariableOp_144¢AssignVariableOp_145¢AssignVariableOp_146¢AssignVariableOp_147¢AssignVariableOp_148¢AssignVariableOp_149¢AssignVariableOp_15¢AssignVariableOp_150¢AssignVariableOp_151¢AssignVariableOp_152¢AssignVariableOp_153¢AssignVariableOp_154¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_99·W
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*ÜV
valueÒVBÏVB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-20/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-20/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-20/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH­
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*Î
valueÄBÁB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ±
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesó
ð::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*­
dtypes¢
2		[
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
:
AssignVariableOp_3AssignVariableOp$assignvariableop_3_dense_1001_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_1001_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp0assignvariableop_5_batch_normalization_904_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_904_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_904_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_8AssignVariableOp:assignvariableop_8_batch_normalization_904_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp$assignvariableop_9_dense_1002_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_1002_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_11AssignVariableOp1assignvariableop_11_batch_normalization_905_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_905_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_905_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_14AssignVariableOp;assignvariableop_14_batch_normalization_905_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp%assignvariableop_15_dense_1003_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_1003_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_17AssignVariableOp1assignvariableop_17_batch_normalization_906_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_906_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_19AssignVariableOp7assignvariableop_19_batch_normalization_906_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_20AssignVariableOp;assignvariableop_20_batch_normalization_906_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp%assignvariableop_21_dense_1004_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp#assignvariableop_22_dense_1004_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_23AssignVariableOp1assignvariableop_23_batch_normalization_907_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_907_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_25AssignVariableOp7assignvariableop_25_batch_normalization_907_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_26AssignVariableOp;assignvariableop_26_batch_normalization_907_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp%assignvariableop_27_dense_1005_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp#assignvariableop_28_dense_1005_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_29AssignVariableOp1assignvariableop_29_batch_normalization_908_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_normalization_908_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_31AssignVariableOp7assignvariableop_31_batch_normalization_908_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_32AssignVariableOp;assignvariableop_32_batch_normalization_908_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp%assignvariableop_33_dense_1006_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp#assignvariableop_34_dense_1006_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_35AssignVariableOp1assignvariableop_35_batch_normalization_909_gammaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_909_betaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_37AssignVariableOp7assignvariableop_37_batch_normalization_909_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_38AssignVariableOp;assignvariableop_38_batch_normalization_909_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp%assignvariableop_39_dense_1007_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp#assignvariableop_40_dense_1007_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_41AssignVariableOp1assignvariableop_41_batch_normalization_910_gammaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_42AssignVariableOp0assignvariableop_42_batch_normalization_910_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_43AssignVariableOp7assignvariableop_43_batch_normalization_910_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_44AssignVariableOp;assignvariableop_44_batch_normalization_910_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp%assignvariableop_45_dense_1008_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp#assignvariableop_46_dense_1008_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_47AssignVariableOp1assignvariableop_47_batch_normalization_911_gammaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_48AssignVariableOp0assignvariableop_48_batch_normalization_911_betaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_49AssignVariableOp7assignvariableop_49_batch_normalization_911_moving_meanIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_50AssignVariableOp;assignvariableop_50_batch_normalization_911_moving_varianceIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp%assignvariableop_51_dense_1009_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp#assignvariableop_52_dense_1009_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_53AssignVariableOp1assignvariableop_53_batch_normalization_912_gammaIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_54AssignVariableOp0assignvariableop_54_batch_normalization_912_betaIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_55AssignVariableOp7assignvariableop_55_batch_normalization_912_moving_meanIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_56AssignVariableOp;assignvariableop_56_batch_normalization_912_moving_varianceIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp%assignvariableop_57_dense_1010_kernelIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp#assignvariableop_58_dense_1010_biasIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_59AssignVariableOp1assignvariableop_59_batch_normalization_913_gammaIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_60AssignVariableOp0assignvariableop_60_batch_normalization_913_betaIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_61AssignVariableOp7assignvariableop_61_batch_normalization_913_moving_meanIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_62AssignVariableOp;assignvariableop_62_batch_normalization_913_moving_varianceIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp%assignvariableop_63_dense_1011_kernelIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp#assignvariableop_64_dense_1011_biasIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_65AssignVariableOpassignvariableop_65_adam_iterIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOpassignvariableop_66_adam_beta_1Identity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOpassignvariableop_67_adam_beta_2Identity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOpassignvariableop_68_adam_decayIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOpassignvariableop_69_totalIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOpassignvariableop_70_count_1Identity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adam_dense_1001_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_dense_1001_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_73AssignVariableOp8assignvariableop_73_adam_batch_normalization_904_gamma_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_74AssignVariableOp7assignvariableop_74_adam_batch_normalization_904_beta_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_75AssignVariableOp,assignvariableop_75_adam_dense_1002_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_76AssignVariableOp*assignvariableop_76_adam_dense_1002_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_77AssignVariableOp8assignvariableop_77_adam_batch_normalization_905_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_78AssignVariableOp7assignvariableop_78_adam_batch_normalization_905_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_79AssignVariableOp,assignvariableop_79_adam_dense_1003_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_80AssignVariableOp*assignvariableop_80_adam_dense_1003_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_81AssignVariableOp8assignvariableop_81_adam_batch_normalization_906_gamma_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_82AssignVariableOp7assignvariableop_82_adam_batch_normalization_906_beta_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_83AssignVariableOp,assignvariableop_83_adam_dense_1004_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_84AssignVariableOp*assignvariableop_84_adam_dense_1004_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_85AssignVariableOp8assignvariableop_85_adam_batch_normalization_907_gamma_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_86AssignVariableOp7assignvariableop_86_adam_batch_normalization_907_beta_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_87AssignVariableOp,assignvariableop_87_adam_dense_1005_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_88AssignVariableOp*assignvariableop_88_adam_dense_1005_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_89AssignVariableOp8assignvariableop_89_adam_batch_normalization_908_gamma_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_90AssignVariableOp7assignvariableop_90_adam_batch_normalization_908_beta_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_91AssignVariableOp,assignvariableop_91_adam_dense_1006_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_92AssignVariableOp*assignvariableop_92_adam_dense_1006_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_93AssignVariableOp8assignvariableop_93_adam_batch_normalization_909_gamma_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_94AssignVariableOp7assignvariableop_94_adam_batch_normalization_909_beta_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_95AssignVariableOp,assignvariableop_95_adam_dense_1007_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_96AssignVariableOp*assignvariableop_96_adam_dense_1007_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_97AssignVariableOp8assignvariableop_97_adam_batch_normalization_910_gamma_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_98AssignVariableOp7assignvariableop_98_adam_batch_normalization_910_beta_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_99AssignVariableOp,assignvariableop_99_adam_dense_1008_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_100AssignVariableOp+assignvariableop_100_adam_dense_1008_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_101AssignVariableOp9assignvariableop_101_adam_batch_normalization_911_gamma_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_102AssignVariableOp8assignvariableop_102_adam_batch_normalization_911_beta_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_103AssignVariableOp-assignvariableop_103_adam_dense_1009_kernel_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_104AssignVariableOp+assignvariableop_104_adam_dense_1009_bias_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_105AssignVariableOp9assignvariableop_105_adam_batch_normalization_912_gamma_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_106AssignVariableOp8assignvariableop_106_adam_batch_normalization_912_beta_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_107AssignVariableOp-assignvariableop_107_adam_dense_1010_kernel_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_108AssignVariableOp+assignvariableop_108_adam_dense_1010_bias_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_109AssignVariableOp9assignvariableop_109_adam_batch_normalization_913_gamma_mIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_110AssignVariableOp8assignvariableop_110_adam_batch_normalization_913_beta_mIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_111AssignVariableOp-assignvariableop_111_adam_dense_1011_kernel_mIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_112AssignVariableOp+assignvariableop_112_adam_dense_1011_bias_mIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_113AssignVariableOp-assignvariableop_113_adam_dense_1001_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_114AssignVariableOp+assignvariableop_114_adam_dense_1001_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_115AssignVariableOp9assignvariableop_115_adam_batch_normalization_904_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_116AssignVariableOp8assignvariableop_116_adam_batch_normalization_904_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_117AssignVariableOp-assignvariableop_117_adam_dense_1002_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_118AssignVariableOp+assignvariableop_118_adam_dense_1002_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_119AssignVariableOp9assignvariableop_119_adam_batch_normalization_905_gamma_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_120AssignVariableOp8assignvariableop_120_adam_batch_normalization_905_beta_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_121AssignVariableOp-assignvariableop_121_adam_dense_1003_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_122AssignVariableOp+assignvariableop_122_adam_dense_1003_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_123AssignVariableOp9assignvariableop_123_adam_batch_normalization_906_gamma_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_124AssignVariableOp8assignvariableop_124_adam_batch_normalization_906_beta_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_125AssignVariableOp-assignvariableop_125_adam_dense_1004_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_126AssignVariableOp+assignvariableop_126_adam_dense_1004_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_127AssignVariableOp9assignvariableop_127_adam_batch_normalization_907_gamma_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_128AssignVariableOp8assignvariableop_128_adam_batch_normalization_907_beta_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_129AssignVariableOp-assignvariableop_129_adam_dense_1005_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_130AssignVariableOp+assignvariableop_130_adam_dense_1005_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_131AssignVariableOp9assignvariableop_131_adam_batch_normalization_908_gamma_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_132AssignVariableOp8assignvariableop_132_adam_batch_normalization_908_beta_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_133AssignVariableOp-assignvariableop_133_adam_dense_1006_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_134AssignVariableOp+assignvariableop_134_adam_dense_1006_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_135AssignVariableOp9assignvariableop_135_adam_batch_normalization_909_gamma_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_136AssignVariableOp8assignvariableop_136_adam_batch_normalization_909_beta_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_137AssignVariableOp-assignvariableop_137_adam_dense_1007_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_138AssignVariableOp+assignvariableop_138_adam_dense_1007_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_139AssignVariableOp9assignvariableop_139_adam_batch_normalization_910_gamma_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_140AssignVariableOp8assignvariableop_140_adam_batch_normalization_910_beta_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_141AssignVariableOp-assignvariableop_141_adam_dense_1008_kernel_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_142AssignVariableOp+assignvariableop_142_adam_dense_1008_bias_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_143AssignVariableOp9assignvariableop_143_adam_batch_normalization_911_gamma_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_144AssignVariableOp8assignvariableop_144_adam_batch_normalization_911_beta_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_145AssignVariableOp-assignvariableop_145_adam_dense_1009_kernel_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_146AssignVariableOp+assignvariableop_146_adam_dense_1009_bias_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_147AssignVariableOp9assignvariableop_147_adam_batch_normalization_912_gamma_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_148AssignVariableOp8assignvariableop_148_adam_batch_normalization_912_beta_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_149AssignVariableOp-assignvariableop_149_adam_dense_1010_kernel_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_150AssignVariableOp+assignvariableop_150_adam_dense_1010_bias_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_151AssignVariableOp9assignvariableop_151_adam_batch_normalization_913_gamma_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_152AssignVariableOp8assignvariableop_152_adam_batch_normalization_913_beta_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_153AssignVariableOp-assignvariableop_153_adam_dense_1011_kernel_vIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_154AssignVariableOp+assignvariableop_154_adam_dense_1011_bias_vIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ù
Identity_155Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_156IdentityIdentity_155:output:0^NoOp_1*
T0*
_output_shapes
: Å
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_156Identity_156:output:0*Í
_input_shapes»
¸: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_154AssignVariableOp_1542*
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
­
Ô
9__inference_batch_normalization_908_layer_call_fn_1003367

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_908_layer_call_and_return_conditional_losses_999796o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1002_layer_call_and_return_conditional_losses_1003027

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
­
Ô
9__inference_batch_normalization_906_layer_call_fn_1003149

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_906_layer_call_and_return_conditional_losses_999632o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_904_layer_call_and_return_conditional_losses_999515

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
È

,__inference_dense_1006_layer_call_fn_1003453

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1006_layer_call_and_return_conditional_losses_1000448o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
Ô
9__inference_batch_normalization_906_layer_call_fn_1003162

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_906_layer_call_and_return_conditional_losses_999679o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
Ô
9__inference_batch_normalization_907_layer_call_fn_1003271

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_907_layer_call_and_return_conditional_losses_999761o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_908_layer_call_and_return_conditional_losses_1003444

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_dense_1011_layer_call_fn_1003998

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÜ
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
GPU 2J 8 *P
fKRI
G__inference_dense_1011_layer_call_and_return_conditional_losses_1000608o
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
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_906_layer_call_and_return_conditional_losses_999632

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_904_layer_call_and_return_conditional_losses_1002998

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
1__inference_leaky_re_lu_905_layer_call_fn_1003112

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
L__inference_leaky_re_lu_905_layer_call_and_return_conditional_losses_1000340`
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
­
Ô
9__inference_batch_normalization_909_layer_call_fn_1003476

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_909_layer_call_and_return_conditional_losses_999878o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®í
ðA
J__inference_sequential_97_layer_call_and_return_conditional_losses_1002717

inputs
normalization_97_sub_y
normalization_97_sqrt_x;
)dense_1001_matmul_readvariableop_resource:58
*dense_1001_biasadd_readvariableop_resource:5M
?batch_normalization_904_assignmovingavg_readvariableop_resource:5O
Abatch_normalization_904_assignmovingavg_1_readvariableop_resource:5K
=batch_normalization_904_batchnorm_mul_readvariableop_resource:5G
9batch_normalization_904_batchnorm_readvariableop_resource:5;
)dense_1002_matmul_readvariableop_resource:558
*dense_1002_biasadd_readvariableop_resource:5M
?batch_normalization_905_assignmovingavg_readvariableop_resource:5O
Abatch_normalization_905_assignmovingavg_1_readvariableop_resource:5K
=batch_normalization_905_batchnorm_mul_readvariableop_resource:5G
9batch_normalization_905_batchnorm_readvariableop_resource:5;
)dense_1003_matmul_readvariableop_resource:58
*dense_1003_biasadd_readvariableop_resource:M
?batch_normalization_906_assignmovingavg_readvariableop_resource:O
Abatch_normalization_906_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_906_batchnorm_mul_readvariableop_resource:G
9batch_normalization_906_batchnorm_readvariableop_resource:;
)dense_1004_matmul_readvariableop_resource:8
*dense_1004_biasadd_readvariableop_resource:M
?batch_normalization_907_assignmovingavg_readvariableop_resource:O
Abatch_normalization_907_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_907_batchnorm_mul_readvariableop_resource:G
9batch_normalization_907_batchnorm_readvariableop_resource:;
)dense_1005_matmul_readvariableop_resource:8
*dense_1005_biasadd_readvariableop_resource:M
?batch_normalization_908_assignmovingavg_readvariableop_resource:O
Abatch_normalization_908_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_908_batchnorm_mul_readvariableop_resource:G
9batch_normalization_908_batchnorm_readvariableop_resource:;
)dense_1006_matmul_readvariableop_resource:8
*dense_1006_biasadd_readvariableop_resource:M
?batch_normalization_909_assignmovingavg_readvariableop_resource:O
Abatch_normalization_909_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_909_batchnorm_mul_readvariableop_resource:G
9batch_normalization_909_batchnorm_readvariableop_resource:;
)dense_1007_matmul_readvariableop_resource:8
*dense_1007_biasadd_readvariableop_resource:M
?batch_normalization_910_assignmovingavg_readvariableop_resource:O
Abatch_normalization_910_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_910_batchnorm_mul_readvariableop_resource:G
9batch_normalization_910_batchnorm_readvariableop_resource:;
)dense_1008_matmul_readvariableop_resource:8
*dense_1008_biasadd_readvariableop_resource:M
?batch_normalization_911_assignmovingavg_readvariableop_resource:O
Abatch_normalization_911_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_911_batchnorm_mul_readvariableop_resource:G
9batch_normalization_911_batchnorm_readvariableop_resource:;
)dense_1009_matmul_readvariableop_resource:8
*dense_1009_biasadd_readvariableop_resource:M
?batch_normalization_912_assignmovingavg_readvariableop_resource:O
Abatch_normalization_912_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_912_batchnorm_mul_readvariableop_resource:G
9batch_normalization_912_batchnorm_readvariableop_resource:;
)dense_1010_matmul_readvariableop_resource:8
*dense_1010_biasadd_readvariableop_resource:M
?batch_normalization_913_assignmovingavg_readvariableop_resource:O
Abatch_normalization_913_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_913_batchnorm_mul_readvariableop_resource:G
9batch_normalization_913_batchnorm_readvariableop_resource:;
)dense_1011_matmul_readvariableop_resource:8
*dense_1011_biasadd_readvariableop_resource:
identity¢'batch_normalization_904/AssignMovingAvg¢6batch_normalization_904/AssignMovingAvg/ReadVariableOp¢)batch_normalization_904/AssignMovingAvg_1¢8batch_normalization_904/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_904/batchnorm/ReadVariableOp¢4batch_normalization_904/batchnorm/mul/ReadVariableOp¢'batch_normalization_905/AssignMovingAvg¢6batch_normalization_905/AssignMovingAvg/ReadVariableOp¢)batch_normalization_905/AssignMovingAvg_1¢8batch_normalization_905/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_905/batchnorm/ReadVariableOp¢4batch_normalization_905/batchnorm/mul/ReadVariableOp¢'batch_normalization_906/AssignMovingAvg¢6batch_normalization_906/AssignMovingAvg/ReadVariableOp¢)batch_normalization_906/AssignMovingAvg_1¢8batch_normalization_906/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_906/batchnorm/ReadVariableOp¢4batch_normalization_906/batchnorm/mul/ReadVariableOp¢'batch_normalization_907/AssignMovingAvg¢6batch_normalization_907/AssignMovingAvg/ReadVariableOp¢)batch_normalization_907/AssignMovingAvg_1¢8batch_normalization_907/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_907/batchnorm/ReadVariableOp¢4batch_normalization_907/batchnorm/mul/ReadVariableOp¢'batch_normalization_908/AssignMovingAvg¢6batch_normalization_908/AssignMovingAvg/ReadVariableOp¢)batch_normalization_908/AssignMovingAvg_1¢8batch_normalization_908/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_908/batchnorm/ReadVariableOp¢4batch_normalization_908/batchnorm/mul/ReadVariableOp¢'batch_normalization_909/AssignMovingAvg¢6batch_normalization_909/AssignMovingAvg/ReadVariableOp¢)batch_normalization_909/AssignMovingAvg_1¢8batch_normalization_909/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_909/batchnorm/ReadVariableOp¢4batch_normalization_909/batchnorm/mul/ReadVariableOp¢'batch_normalization_910/AssignMovingAvg¢6batch_normalization_910/AssignMovingAvg/ReadVariableOp¢)batch_normalization_910/AssignMovingAvg_1¢8batch_normalization_910/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_910/batchnorm/ReadVariableOp¢4batch_normalization_910/batchnorm/mul/ReadVariableOp¢'batch_normalization_911/AssignMovingAvg¢6batch_normalization_911/AssignMovingAvg/ReadVariableOp¢)batch_normalization_911/AssignMovingAvg_1¢8batch_normalization_911/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_911/batchnorm/ReadVariableOp¢4batch_normalization_911/batchnorm/mul/ReadVariableOp¢'batch_normalization_912/AssignMovingAvg¢6batch_normalization_912/AssignMovingAvg/ReadVariableOp¢)batch_normalization_912/AssignMovingAvg_1¢8batch_normalization_912/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_912/batchnorm/ReadVariableOp¢4batch_normalization_912/batchnorm/mul/ReadVariableOp¢'batch_normalization_913/AssignMovingAvg¢6batch_normalization_913/AssignMovingAvg/ReadVariableOp¢)batch_normalization_913/AssignMovingAvg_1¢8batch_normalization_913/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_913/batchnorm/ReadVariableOp¢4batch_normalization_913/batchnorm/mul/ReadVariableOp¢!dense_1001/BiasAdd/ReadVariableOp¢ dense_1001/MatMul/ReadVariableOp¢!dense_1002/BiasAdd/ReadVariableOp¢ dense_1002/MatMul/ReadVariableOp¢!dense_1003/BiasAdd/ReadVariableOp¢ dense_1003/MatMul/ReadVariableOp¢!dense_1004/BiasAdd/ReadVariableOp¢ dense_1004/MatMul/ReadVariableOp¢!dense_1005/BiasAdd/ReadVariableOp¢ dense_1005/MatMul/ReadVariableOp¢!dense_1006/BiasAdd/ReadVariableOp¢ dense_1006/MatMul/ReadVariableOp¢!dense_1007/BiasAdd/ReadVariableOp¢ dense_1007/MatMul/ReadVariableOp¢!dense_1008/BiasAdd/ReadVariableOp¢ dense_1008/MatMul/ReadVariableOp¢!dense_1009/BiasAdd/ReadVariableOp¢ dense_1009/MatMul/ReadVariableOp¢!dense_1010/BiasAdd/ReadVariableOp¢ dense_1010/MatMul/ReadVariableOp¢!dense_1011/BiasAdd/ReadVariableOp¢ dense_1011/MatMul/ReadVariableOpm
normalization_97/subSubinputsnormalization_97_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_97/SqrtSqrtnormalization_97_sqrt_x*
T0*
_output_shapes

:_
normalization_97/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_97/MaximumMaximumnormalization_97/Sqrt:y:0#normalization_97/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_97/truedivRealDivnormalization_97/sub:z:0normalization_97/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_1001/MatMul/ReadVariableOpReadVariableOp)dense_1001_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0
dense_1001/MatMulMatMulnormalization_97/truediv:z:0(dense_1001/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
!dense_1001/BiasAdd/ReadVariableOpReadVariableOp*dense_1001_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_1001/BiasAddBiasAdddense_1001/MatMul:product:0)dense_1001/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
6batch_normalization_904/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ä
$batch_normalization_904/moments/meanMeandense_1001/BiasAdd:output:0?batch_normalization_904/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(
,batch_normalization_904/moments/StopGradientStopGradient-batch_normalization_904/moments/mean:output:0*
T0*
_output_shapes

:5Ì
1batch_normalization_904/moments/SquaredDifferenceSquaredDifferencedense_1001/BiasAdd:output:05batch_normalization_904/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
:batch_normalization_904/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_904/moments/varianceMean5batch_normalization_904/moments/SquaredDifference:z:0Cbatch_normalization_904/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(
'batch_normalization_904/moments/SqueezeSqueeze-batch_normalization_904/moments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 £
)batch_normalization_904/moments/Squeeze_1Squeeze1batch_normalization_904/moments/variance:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 r
-batch_normalization_904/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_904/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_904_assignmovingavg_readvariableop_resource*
_output_shapes
:5*
dtype0É
+batch_normalization_904/AssignMovingAvg/subSub>batch_normalization_904/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_904/moments/Squeeze:output:0*
T0*
_output_shapes
:5À
+batch_normalization_904/AssignMovingAvg/mulMul/batch_normalization_904/AssignMovingAvg/sub:z:06batch_normalization_904/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5
'batch_normalization_904/AssignMovingAvgAssignSubVariableOp?batch_normalization_904_assignmovingavg_readvariableop_resource/batch_normalization_904/AssignMovingAvg/mul:z:07^batch_normalization_904/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_904/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_904/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_904_assignmovingavg_1_readvariableop_resource*
_output_shapes
:5*
dtype0Ï
-batch_normalization_904/AssignMovingAvg_1/subSub@batch_normalization_904/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_904/moments/Squeeze_1:output:0*
T0*
_output_shapes
:5Æ
-batch_normalization_904/AssignMovingAvg_1/mulMul1batch_normalization_904/AssignMovingAvg_1/sub:z:08batch_normalization_904/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5
)batch_normalization_904/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_904_assignmovingavg_1_readvariableop_resource1batch_normalization_904/AssignMovingAvg_1/mul:z:09^batch_normalization_904/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_904/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_904/batchnorm/addAddV22batch_normalization_904/moments/Squeeze_1:output:00batch_normalization_904/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
'batch_normalization_904/batchnorm/RsqrtRsqrt)batch_normalization_904/batchnorm/add:z:0*
T0*
_output_shapes
:5®
4batch_normalization_904/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_904_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0¼
%batch_normalization_904/batchnorm/mulMul+batch_normalization_904/batchnorm/Rsqrt:y:0<batch_normalization_904/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5¨
'batch_normalization_904/batchnorm/mul_1Muldense_1001/BiasAdd:output:0)batch_normalization_904/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5°
'batch_normalization_904/batchnorm/mul_2Mul0batch_normalization_904/moments/Squeeze:output:0)batch_normalization_904/batchnorm/mul:z:0*
T0*
_output_shapes
:5¦
0batch_normalization_904/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_904_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0¸
%batch_normalization_904/batchnorm/subSub8batch_normalization_904/batchnorm/ReadVariableOp:value:0+batch_normalization_904/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5º
'batch_normalization_904/batchnorm/add_1AddV2+batch_normalization_904/batchnorm/mul_1:z:0)batch_normalization_904/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
leaky_re_lu_904/LeakyRelu	LeakyRelu+batch_normalization_904/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>
 dense_1002/MatMul/ReadVariableOpReadVariableOp)dense_1002_matmul_readvariableop_resource*
_output_shapes

:55*
dtype0 
dense_1002/MatMulMatMul'leaky_re_lu_904/LeakyRelu:activations:0(dense_1002/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
!dense_1002/BiasAdd/ReadVariableOpReadVariableOp*dense_1002_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_1002/BiasAddBiasAdddense_1002/MatMul:product:0)dense_1002/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
6batch_normalization_905/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ä
$batch_normalization_905/moments/meanMeandense_1002/BiasAdd:output:0?batch_normalization_905/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(
,batch_normalization_905/moments/StopGradientStopGradient-batch_normalization_905/moments/mean:output:0*
T0*
_output_shapes

:5Ì
1batch_normalization_905/moments/SquaredDifferenceSquaredDifferencedense_1002/BiasAdd:output:05batch_normalization_905/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
:batch_normalization_905/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_905/moments/varianceMean5batch_normalization_905/moments/SquaredDifference:z:0Cbatch_normalization_905/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:5*
	keep_dims(
'batch_normalization_905/moments/SqueezeSqueeze-batch_normalization_905/moments/mean:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 £
)batch_normalization_905/moments/Squeeze_1Squeeze1batch_normalization_905/moments/variance:output:0*
T0*
_output_shapes
:5*
squeeze_dims
 r
-batch_normalization_905/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_905/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_905_assignmovingavg_readvariableop_resource*
_output_shapes
:5*
dtype0É
+batch_normalization_905/AssignMovingAvg/subSub>batch_normalization_905/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_905/moments/Squeeze:output:0*
T0*
_output_shapes
:5À
+batch_normalization_905/AssignMovingAvg/mulMul/batch_normalization_905/AssignMovingAvg/sub:z:06batch_normalization_905/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:5
'batch_normalization_905/AssignMovingAvgAssignSubVariableOp?batch_normalization_905_assignmovingavg_readvariableop_resource/batch_normalization_905/AssignMovingAvg/mul:z:07^batch_normalization_905/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_905/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_905/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_905_assignmovingavg_1_readvariableop_resource*
_output_shapes
:5*
dtype0Ï
-batch_normalization_905/AssignMovingAvg_1/subSub@batch_normalization_905/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_905/moments/Squeeze_1:output:0*
T0*
_output_shapes
:5Æ
-batch_normalization_905/AssignMovingAvg_1/mulMul1batch_normalization_905/AssignMovingAvg_1/sub:z:08batch_normalization_905/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:5
)batch_normalization_905/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_905_assignmovingavg_1_readvariableop_resource1batch_normalization_905/AssignMovingAvg_1/mul:z:09^batch_normalization_905/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_905/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_905/batchnorm/addAddV22batch_normalization_905/moments/Squeeze_1:output:00batch_normalization_905/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
'batch_normalization_905/batchnorm/RsqrtRsqrt)batch_normalization_905/batchnorm/add:z:0*
T0*
_output_shapes
:5®
4batch_normalization_905/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_905_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0¼
%batch_normalization_905/batchnorm/mulMul+batch_normalization_905/batchnorm/Rsqrt:y:0<batch_normalization_905/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5¨
'batch_normalization_905/batchnorm/mul_1Muldense_1002/BiasAdd:output:0)batch_normalization_905/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5°
'batch_normalization_905/batchnorm/mul_2Mul0batch_normalization_905/moments/Squeeze:output:0)batch_normalization_905/batchnorm/mul:z:0*
T0*
_output_shapes
:5¦
0batch_normalization_905/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_905_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0¸
%batch_normalization_905/batchnorm/subSub8batch_normalization_905/batchnorm/ReadVariableOp:value:0+batch_normalization_905/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5º
'batch_normalization_905/batchnorm/add_1AddV2+batch_normalization_905/batchnorm/mul_1:z:0)batch_normalization_905/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
leaky_re_lu_905/LeakyRelu	LeakyRelu+batch_normalization_905/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>
 dense_1003/MatMul/ReadVariableOpReadVariableOp)dense_1003_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0 
dense_1003/MatMulMatMul'leaky_re_lu_905/LeakyRelu:activations:0(dense_1003/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1003/BiasAdd/ReadVariableOpReadVariableOp*dense_1003_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1003/BiasAddBiasAdddense_1003/MatMul:product:0)dense_1003/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_906/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ä
$batch_normalization_906/moments/meanMeandense_1003/BiasAdd:output:0?batch_normalization_906/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_906/moments/StopGradientStopGradient-batch_normalization_906/moments/mean:output:0*
T0*
_output_shapes

:Ì
1batch_normalization_906/moments/SquaredDifferenceSquaredDifferencedense_1003/BiasAdd:output:05batch_normalization_906/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_906/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_906/moments/varianceMean5batch_normalization_906/moments/SquaredDifference:z:0Cbatch_normalization_906/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_906/moments/SqueezeSqueeze-batch_normalization_906/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_906/moments/Squeeze_1Squeeze1batch_normalization_906/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_906/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_906/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_906_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_906/AssignMovingAvg/subSub>batch_normalization_906/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_906/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_906/AssignMovingAvg/mulMul/batch_normalization_906/AssignMovingAvg/sub:z:06batch_normalization_906/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_906/AssignMovingAvgAssignSubVariableOp?batch_normalization_906_assignmovingavg_readvariableop_resource/batch_normalization_906/AssignMovingAvg/mul:z:07^batch_normalization_906/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_906/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_906/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_906_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_906/AssignMovingAvg_1/subSub@batch_normalization_906/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_906/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_906/AssignMovingAvg_1/mulMul1batch_normalization_906/AssignMovingAvg_1/sub:z:08batch_normalization_906/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_906/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_906_assignmovingavg_1_readvariableop_resource1batch_normalization_906/AssignMovingAvg_1/mul:z:09^batch_normalization_906/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_906/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_906/batchnorm/addAddV22batch_normalization_906/moments/Squeeze_1:output:00batch_normalization_906/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_906/batchnorm/RsqrtRsqrt)batch_normalization_906/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_906/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_906_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_906/batchnorm/mulMul+batch_normalization_906/batchnorm/Rsqrt:y:0<batch_normalization_906/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_906/batchnorm/mul_1Muldense_1003/BiasAdd:output:0)batch_normalization_906/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_906/batchnorm/mul_2Mul0batch_normalization_906/moments/Squeeze:output:0)batch_normalization_906/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_906/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_906_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_906/batchnorm/subSub8batch_normalization_906/batchnorm/ReadVariableOp:value:0+batch_normalization_906/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_906/batchnorm/add_1AddV2+batch_normalization_906/batchnorm/mul_1:z:0)batch_normalization_906/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_906/LeakyRelu	LeakyRelu+batch_normalization_906/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1004/MatMul/ReadVariableOpReadVariableOp)dense_1004_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1004/MatMulMatMul'leaky_re_lu_906/LeakyRelu:activations:0(dense_1004/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1004/BiasAdd/ReadVariableOpReadVariableOp*dense_1004_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1004/BiasAddBiasAdddense_1004/MatMul:product:0)dense_1004/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_907/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ä
$batch_normalization_907/moments/meanMeandense_1004/BiasAdd:output:0?batch_normalization_907/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_907/moments/StopGradientStopGradient-batch_normalization_907/moments/mean:output:0*
T0*
_output_shapes

:Ì
1batch_normalization_907/moments/SquaredDifferenceSquaredDifferencedense_1004/BiasAdd:output:05batch_normalization_907/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_907/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_907/moments/varianceMean5batch_normalization_907/moments/SquaredDifference:z:0Cbatch_normalization_907/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_907/moments/SqueezeSqueeze-batch_normalization_907/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_907/moments/Squeeze_1Squeeze1batch_normalization_907/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_907/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_907/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_907_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_907/AssignMovingAvg/subSub>batch_normalization_907/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_907/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_907/AssignMovingAvg/mulMul/batch_normalization_907/AssignMovingAvg/sub:z:06batch_normalization_907/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_907/AssignMovingAvgAssignSubVariableOp?batch_normalization_907_assignmovingavg_readvariableop_resource/batch_normalization_907/AssignMovingAvg/mul:z:07^batch_normalization_907/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_907/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_907/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_907_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_907/AssignMovingAvg_1/subSub@batch_normalization_907/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_907/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_907/AssignMovingAvg_1/mulMul1batch_normalization_907/AssignMovingAvg_1/sub:z:08batch_normalization_907/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_907/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_907_assignmovingavg_1_readvariableop_resource1batch_normalization_907/AssignMovingAvg_1/mul:z:09^batch_normalization_907/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_907/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_907/batchnorm/addAddV22batch_normalization_907/moments/Squeeze_1:output:00batch_normalization_907/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_907/batchnorm/RsqrtRsqrt)batch_normalization_907/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_907/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_907_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_907/batchnorm/mulMul+batch_normalization_907/batchnorm/Rsqrt:y:0<batch_normalization_907/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_907/batchnorm/mul_1Muldense_1004/BiasAdd:output:0)batch_normalization_907/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_907/batchnorm/mul_2Mul0batch_normalization_907/moments/Squeeze:output:0)batch_normalization_907/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_907/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_907_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_907/batchnorm/subSub8batch_normalization_907/batchnorm/ReadVariableOp:value:0+batch_normalization_907/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_907/batchnorm/add_1AddV2+batch_normalization_907/batchnorm/mul_1:z:0)batch_normalization_907/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_907/LeakyRelu	LeakyRelu+batch_normalization_907/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1005/MatMul/ReadVariableOpReadVariableOp)dense_1005_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1005/MatMulMatMul'leaky_re_lu_907/LeakyRelu:activations:0(dense_1005/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1005/BiasAdd/ReadVariableOpReadVariableOp*dense_1005_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1005/BiasAddBiasAdddense_1005/MatMul:product:0)dense_1005/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_908/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ä
$batch_normalization_908/moments/meanMeandense_1005/BiasAdd:output:0?batch_normalization_908/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_908/moments/StopGradientStopGradient-batch_normalization_908/moments/mean:output:0*
T0*
_output_shapes

:Ì
1batch_normalization_908/moments/SquaredDifferenceSquaredDifferencedense_1005/BiasAdd:output:05batch_normalization_908/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_908/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_908/moments/varianceMean5batch_normalization_908/moments/SquaredDifference:z:0Cbatch_normalization_908/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_908/moments/SqueezeSqueeze-batch_normalization_908/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_908/moments/Squeeze_1Squeeze1batch_normalization_908/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_908/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_908/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_908_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_908/AssignMovingAvg/subSub>batch_normalization_908/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_908/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_908/AssignMovingAvg/mulMul/batch_normalization_908/AssignMovingAvg/sub:z:06batch_normalization_908/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_908/AssignMovingAvgAssignSubVariableOp?batch_normalization_908_assignmovingavg_readvariableop_resource/batch_normalization_908/AssignMovingAvg/mul:z:07^batch_normalization_908/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_908/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_908/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_908_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_908/AssignMovingAvg_1/subSub@batch_normalization_908/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_908/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_908/AssignMovingAvg_1/mulMul1batch_normalization_908/AssignMovingAvg_1/sub:z:08batch_normalization_908/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_908/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_908_assignmovingavg_1_readvariableop_resource1batch_normalization_908/AssignMovingAvg_1/mul:z:09^batch_normalization_908/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_908/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_908/batchnorm/addAddV22batch_normalization_908/moments/Squeeze_1:output:00batch_normalization_908/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_908/batchnorm/RsqrtRsqrt)batch_normalization_908/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_908/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_908_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_908/batchnorm/mulMul+batch_normalization_908/batchnorm/Rsqrt:y:0<batch_normalization_908/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_908/batchnorm/mul_1Muldense_1005/BiasAdd:output:0)batch_normalization_908/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_908/batchnorm/mul_2Mul0batch_normalization_908/moments/Squeeze:output:0)batch_normalization_908/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_908/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_908_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_908/batchnorm/subSub8batch_normalization_908/batchnorm/ReadVariableOp:value:0+batch_normalization_908/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_908/batchnorm/add_1AddV2+batch_normalization_908/batchnorm/mul_1:z:0)batch_normalization_908/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_908/LeakyRelu	LeakyRelu+batch_normalization_908/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1006/MatMul/ReadVariableOpReadVariableOp)dense_1006_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1006/MatMulMatMul'leaky_re_lu_908/LeakyRelu:activations:0(dense_1006/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1006/BiasAdd/ReadVariableOpReadVariableOp*dense_1006_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1006/BiasAddBiasAdddense_1006/MatMul:product:0)dense_1006/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_909/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ä
$batch_normalization_909/moments/meanMeandense_1006/BiasAdd:output:0?batch_normalization_909/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_909/moments/StopGradientStopGradient-batch_normalization_909/moments/mean:output:0*
T0*
_output_shapes

:Ì
1batch_normalization_909/moments/SquaredDifferenceSquaredDifferencedense_1006/BiasAdd:output:05batch_normalization_909/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_909/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_909/moments/varianceMean5batch_normalization_909/moments/SquaredDifference:z:0Cbatch_normalization_909/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_909/moments/SqueezeSqueeze-batch_normalization_909/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_909/moments/Squeeze_1Squeeze1batch_normalization_909/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_909/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_909/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_909_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_909/AssignMovingAvg/subSub>batch_normalization_909/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_909/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_909/AssignMovingAvg/mulMul/batch_normalization_909/AssignMovingAvg/sub:z:06batch_normalization_909/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_909/AssignMovingAvgAssignSubVariableOp?batch_normalization_909_assignmovingavg_readvariableop_resource/batch_normalization_909/AssignMovingAvg/mul:z:07^batch_normalization_909/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_909/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_909/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_909_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_909/AssignMovingAvg_1/subSub@batch_normalization_909/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_909/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_909/AssignMovingAvg_1/mulMul1batch_normalization_909/AssignMovingAvg_1/sub:z:08batch_normalization_909/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_909/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_909_assignmovingavg_1_readvariableop_resource1batch_normalization_909/AssignMovingAvg_1/mul:z:09^batch_normalization_909/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_909/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_909/batchnorm/addAddV22batch_normalization_909/moments/Squeeze_1:output:00batch_normalization_909/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_909/batchnorm/RsqrtRsqrt)batch_normalization_909/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_909/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_909_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_909/batchnorm/mulMul+batch_normalization_909/batchnorm/Rsqrt:y:0<batch_normalization_909/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_909/batchnorm/mul_1Muldense_1006/BiasAdd:output:0)batch_normalization_909/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_909/batchnorm/mul_2Mul0batch_normalization_909/moments/Squeeze:output:0)batch_normalization_909/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_909/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_909_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_909/batchnorm/subSub8batch_normalization_909/batchnorm/ReadVariableOp:value:0+batch_normalization_909/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_909/batchnorm/add_1AddV2+batch_normalization_909/batchnorm/mul_1:z:0)batch_normalization_909/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_909/LeakyRelu	LeakyRelu+batch_normalization_909/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1007/MatMul/ReadVariableOpReadVariableOp)dense_1007_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1007/MatMulMatMul'leaky_re_lu_909/LeakyRelu:activations:0(dense_1007/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1007/BiasAdd/ReadVariableOpReadVariableOp*dense_1007_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1007/BiasAddBiasAdddense_1007/MatMul:product:0)dense_1007/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_910/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ä
$batch_normalization_910/moments/meanMeandense_1007/BiasAdd:output:0?batch_normalization_910/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_910/moments/StopGradientStopGradient-batch_normalization_910/moments/mean:output:0*
T0*
_output_shapes

:Ì
1batch_normalization_910/moments/SquaredDifferenceSquaredDifferencedense_1007/BiasAdd:output:05batch_normalization_910/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_910/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_910/moments/varianceMean5batch_normalization_910/moments/SquaredDifference:z:0Cbatch_normalization_910/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_910/moments/SqueezeSqueeze-batch_normalization_910/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_910/moments/Squeeze_1Squeeze1batch_normalization_910/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_910/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_910/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_910_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_910/AssignMovingAvg/subSub>batch_normalization_910/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_910/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_910/AssignMovingAvg/mulMul/batch_normalization_910/AssignMovingAvg/sub:z:06batch_normalization_910/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_910/AssignMovingAvgAssignSubVariableOp?batch_normalization_910_assignmovingavg_readvariableop_resource/batch_normalization_910/AssignMovingAvg/mul:z:07^batch_normalization_910/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_910/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_910/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_910_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_910/AssignMovingAvg_1/subSub@batch_normalization_910/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_910/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_910/AssignMovingAvg_1/mulMul1batch_normalization_910/AssignMovingAvg_1/sub:z:08batch_normalization_910/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_910/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_910_assignmovingavg_1_readvariableop_resource1batch_normalization_910/AssignMovingAvg_1/mul:z:09^batch_normalization_910/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_910/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_910/batchnorm/addAddV22batch_normalization_910/moments/Squeeze_1:output:00batch_normalization_910/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_910/batchnorm/RsqrtRsqrt)batch_normalization_910/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_910/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_910_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_910/batchnorm/mulMul+batch_normalization_910/batchnorm/Rsqrt:y:0<batch_normalization_910/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_910/batchnorm/mul_1Muldense_1007/BiasAdd:output:0)batch_normalization_910/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_910/batchnorm/mul_2Mul0batch_normalization_910/moments/Squeeze:output:0)batch_normalization_910/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_910/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_910_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_910/batchnorm/subSub8batch_normalization_910/batchnorm/ReadVariableOp:value:0+batch_normalization_910/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_910/batchnorm/add_1AddV2+batch_normalization_910/batchnorm/mul_1:z:0)batch_normalization_910/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_910/LeakyRelu	LeakyRelu+batch_normalization_910/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1008/MatMul/ReadVariableOpReadVariableOp)dense_1008_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1008/MatMulMatMul'leaky_re_lu_910/LeakyRelu:activations:0(dense_1008/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1008/BiasAdd/ReadVariableOpReadVariableOp*dense_1008_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1008/BiasAddBiasAdddense_1008/MatMul:product:0)dense_1008/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_911/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ä
$batch_normalization_911/moments/meanMeandense_1008/BiasAdd:output:0?batch_normalization_911/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_911/moments/StopGradientStopGradient-batch_normalization_911/moments/mean:output:0*
T0*
_output_shapes

:Ì
1batch_normalization_911/moments/SquaredDifferenceSquaredDifferencedense_1008/BiasAdd:output:05batch_normalization_911/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_911/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_911/moments/varianceMean5batch_normalization_911/moments/SquaredDifference:z:0Cbatch_normalization_911/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_911/moments/SqueezeSqueeze-batch_normalization_911/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_911/moments/Squeeze_1Squeeze1batch_normalization_911/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_911/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_911/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_911_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_911/AssignMovingAvg/subSub>batch_normalization_911/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_911/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_911/AssignMovingAvg/mulMul/batch_normalization_911/AssignMovingAvg/sub:z:06batch_normalization_911/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_911/AssignMovingAvgAssignSubVariableOp?batch_normalization_911_assignmovingavg_readvariableop_resource/batch_normalization_911/AssignMovingAvg/mul:z:07^batch_normalization_911/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_911/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_911/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_911_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_911/AssignMovingAvg_1/subSub@batch_normalization_911/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_911/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_911/AssignMovingAvg_1/mulMul1batch_normalization_911/AssignMovingAvg_1/sub:z:08batch_normalization_911/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_911/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_911_assignmovingavg_1_readvariableop_resource1batch_normalization_911/AssignMovingAvg_1/mul:z:09^batch_normalization_911/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_911/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_911/batchnorm/addAddV22batch_normalization_911/moments/Squeeze_1:output:00batch_normalization_911/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_911/batchnorm/RsqrtRsqrt)batch_normalization_911/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_911/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_911_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_911/batchnorm/mulMul+batch_normalization_911/batchnorm/Rsqrt:y:0<batch_normalization_911/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_911/batchnorm/mul_1Muldense_1008/BiasAdd:output:0)batch_normalization_911/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_911/batchnorm/mul_2Mul0batch_normalization_911/moments/Squeeze:output:0)batch_normalization_911/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_911/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_911_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_911/batchnorm/subSub8batch_normalization_911/batchnorm/ReadVariableOp:value:0+batch_normalization_911/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_911/batchnorm/add_1AddV2+batch_normalization_911/batchnorm/mul_1:z:0)batch_normalization_911/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_911/LeakyRelu	LeakyRelu+batch_normalization_911/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1009/MatMul/ReadVariableOpReadVariableOp)dense_1009_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1009/MatMulMatMul'leaky_re_lu_911/LeakyRelu:activations:0(dense_1009/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1009/BiasAdd/ReadVariableOpReadVariableOp*dense_1009_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1009/BiasAddBiasAdddense_1009/MatMul:product:0)dense_1009/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_912/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ä
$batch_normalization_912/moments/meanMeandense_1009/BiasAdd:output:0?batch_normalization_912/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_912/moments/StopGradientStopGradient-batch_normalization_912/moments/mean:output:0*
T0*
_output_shapes

:Ì
1batch_normalization_912/moments/SquaredDifferenceSquaredDifferencedense_1009/BiasAdd:output:05batch_normalization_912/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_912/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_912/moments/varianceMean5batch_normalization_912/moments/SquaredDifference:z:0Cbatch_normalization_912/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_912/moments/SqueezeSqueeze-batch_normalization_912/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_912/moments/Squeeze_1Squeeze1batch_normalization_912/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_912/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_912/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_912_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_912/AssignMovingAvg/subSub>batch_normalization_912/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_912/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_912/AssignMovingAvg/mulMul/batch_normalization_912/AssignMovingAvg/sub:z:06batch_normalization_912/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_912/AssignMovingAvgAssignSubVariableOp?batch_normalization_912_assignmovingavg_readvariableop_resource/batch_normalization_912/AssignMovingAvg/mul:z:07^batch_normalization_912/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_912/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_912/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_912_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_912/AssignMovingAvg_1/subSub@batch_normalization_912/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_912/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_912/AssignMovingAvg_1/mulMul1batch_normalization_912/AssignMovingAvg_1/sub:z:08batch_normalization_912/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_912/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_912_assignmovingavg_1_readvariableop_resource1batch_normalization_912/AssignMovingAvg_1/mul:z:09^batch_normalization_912/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_912/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_912/batchnorm/addAddV22batch_normalization_912/moments/Squeeze_1:output:00batch_normalization_912/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_912/batchnorm/RsqrtRsqrt)batch_normalization_912/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_912/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_912_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_912/batchnorm/mulMul+batch_normalization_912/batchnorm/Rsqrt:y:0<batch_normalization_912/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_912/batchnorm/mul_1Muldense_1009/BiasAdd:output:0)batch_normalization_912/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_912/batchnorm/mul_2Mul0batch_normalization_912/moments/Squeeze:output:0)batch_normalization_912/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_912/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_912_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_912/batchnorm/subSub8batch_normalization_912/batchnorm/ReadVariableOp:value:0+batch_normalization_912/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_912/batchnorm/add_1AddV2+batch_normalization_912/batchnorm/mul_1:z:0)batch_normalization_912/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_912/LeakyRelu	LeakyRelu+batch_normalization_912/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1010/MatMul/ReadVariableOpReadVariableOp)dense_1010_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1010/MatMulMatMul'leaky_re_lu_912/LeakyRelu:activations:0(dense_1010/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1010/BiasAdd/ReadVariableOpReadVariableOp*dense_1010_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1010/BiasAddBiasAdddense_1010/MatMul:product:0)dense_1010/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_913/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ä
$batch_normalization_913/moments/meanMeandense_1010/BiasAdd:output:0?batch_normalization_913/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_913/moments/StopGradientStopGradient-batch_normalization_913/moments/mean:output:0*
T0*
_output_shapes

:Ì
1batch_normalization_913/moments/SquaredDifferenceSquaredDifferencedense_1010/BiasAdd:output:05batch_normalization_913/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_913/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_913/moments/varianceMean5batch_normalization_913/moments/SquaredDifference:z:0Cbatch_normalization_913/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_913/moments/SqueezeSqueeze-batch_normalization_913/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_913/moments/Squeeze_1Squeeze1batch_normalization_913/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_913/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_913/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_913_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_913/AssignMovingAvg/subSub>batch_normalization_913/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_913/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_913/AssignMovingAvg/mulMul/batch_normalization_913/AssignMovingAvg/sub:z:06batch_normalization_913/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_913/AssignMovingAvgAssignSubVariableOp?batch_normalization_913_assignmovingavg_readvariableop_resource/batch_normalization_913/AssignMovingAvg/mul:z:07^batch_normalization_913/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_913/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_913/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_913_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_913/AssignMovingAvg_1/subSub@batch_normalization_913/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_913/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_913/AssignMovingAvg_1/mulMul1batch_normalization_913/AssignMovingAvg_1/sub:z:08batch_normalization_913/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_913/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_913_assignmovingavg_1_readvariableop_resource1batch_normalization_913/AssignMovingAvg_1/mul:z:09^batch_normalization_913/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_913/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_913/batchnorm/addAddV22batch_normalization_913/moments/Squeeze_1:output:00batch_normalization_913/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_913/batchnorm/RsqrtRsqrt)batch_normalization_913/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_913/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_913_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_913/batchnorm/mulMul+batch_normalization_913/batchnorm/Rsqrt:y:0<batch_normalization_913/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_913/batchnorm/mul_1Muldense_1010/BiasAdd:output:0)batch_normalization_913/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_913/batchnorm/mul_2Mul0batch_normalization_913/moments/Squeeze:output:0)batch_normalization_913/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_913/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_913_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_913/batchnorm/subSub8batch_normalization_913/batchnorm/ReadVariableOp:value:0+batch_normalization_913/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_913/batchnorm/add_1AddV2+batch_normalization_913/batchnorm/mul_1:z:0)batch_normalization_913/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_913/LeakyRelu	LeakyRelu+batch_normalization_913/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1011/MatMul/ReadVariableOpReadVariableOp)dense_1011_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1011/MatMulMatMul'leaky_re_lu_913/LeakyRelu:activations:0(dense_1011/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1011/BiasAdd/ReadVariableOpReadVariableOp*dense_1011_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1011/BiasAddBiasAdddense_1011/MatMul:product:0)dense_1011/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitydense_1011/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
NoOpNoOp(^batch_normalization_904/AssignMovingAvg7^batch_normalization_904/AssignMovingAvg/ReadVariableOp*^batch_normalization_904/AssignMovingAvg_19^batch_normalization_904/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_904/batchnorm/ReadVariableOp5^batch_normalization_904/batchnorm/mul/ReadVariableOp(^batch_normalization_905/AssignMovingAvg7^batch_normalization_905/AssignMovingAvg/ReadVariableOp*^batch_normalization_905/AssignMovingAvg_19^batch_normalization_905/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_905/batchnorm/ReadVariableOp5^batch_normalization_905/batchnorm/mul/ReadVariableOp(^batch_normalization_906/AssignMovingAvg7^batch_normalization_906/AssignMovingAvg/ReadVariableOp*^batch_normalization_906/AssignMovingAvg_19^batch_normalization_906/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_906/batchnorm/ReadVariableOp5^batch_normalization_906/batchnorm/mul/ReadVariableOp(^batch_normalization_907/AssignMovingAvg7^batch_normalization_907/AssignMovingAvg/ReadVariableOp*^batch_normalization_907/AssignMovingAvg_19^batch_normalization_907/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_907/batchnorm/ReadVariableOp5^batch_normalization_907/batchnorm/mul/ReadVariableOp(^batch_normalization_908/AssignMovingAvg7^batch_normalization_908/AssignMovingAvg/ReadVariableOp*^batch_normalization_908/AssignMovingAvg_19^batch_normalization_908/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_908/batchnorm/ReadVariableOp5^batch_normalization_908/batchnorm/mul/ReadVariableOp(^batch_normalization_909/AssignMovingAvg7^batch_normalization_909/AssignMovingAvg/ReadVariableOp*^batch_normalization_909/AssignMovingAvg_19^batch_normalization_909/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_909/batchnorm/ReadVariableOp5^batch_normalization_909/batchnorm/mul/ReadVariableOp(^batch_normalization_910/AssignMovingAvg7^batch_normalization_910/AssignMovingAvg/ReadVariableOp*^batch_normalization_910/AssignMovingAvg_19^batch_normalization_910/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_910/batchnorm/ReadVariableOp5^batch_normalization_910/batchnorm/mul/ReadVariableOp(^batch_normalization_911/AssignMovingAvg7^batch_normalization_911/AssignMovingAvg/ReadVariableOp*^batch_normalization_911/AssignMovingAvg_19^batch_normalization_911/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_911/batchnorm/ReadVariableOp5^batch_normalization_911/batchnorm/mul/ReadVariableOp(^batch_normalization_912/AssignMovingAvg7^batch_normalization_912/AssignMovingAvg/ReadVariableOp*^batch_normalization_912/AssignMovingAvg_19^batch_normalization_912/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_912/batchnorm/ReadVariableOp5^batch_normalization_912/batchnorm/mul/ReadVariableOp(^batch_normalization_913/AssignMovingAvg7^batch_normalization_913/AssignMovingAvg/ReadVariableOp*^batch_normalization_913/AssignMovingAvg_19^batch_normalization_913/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_913/batchnorm/ReadVariableOp5^batch_normalization_913/batchnorm/mul/ReadVariableOp"^dense_1001/BiasAdd/ReadVariableOp!^dense_1001/MatMul/ReadVariableOp"^dense_1002/BiasAdd/ReadVariableOp!^dense_1002/MatMul/ReadVariableOp"^dense_1003/BiasAdd/ReadVariableOp!^dense_1003/MatMul/ReadVariableOp"^dense_1004/BiasAdd/ReadVariableOp!^dense_1004/MatMul/ReadVariableOp"^dense_1005/BiasAdd/ReadVariableOp!^dense_1005/MatMul/ReadVariableOp"^dense_1006/BiasAdd/ReadVariableOp!^dense_1006/MatMul/ReadVariableOp"^dense_1007/BiasAdd/ReadVariableOp!^dense_1007/MatMul/ReadVariableOp"^dense_1008/BiasAdd/ReadVariableOp!^dense_1008/MatMul/ReadVariableOp"^dense_1009/BiasAdd/ReadVariableOp!^dense_1009/MatMul/ReadVariableOp"^dense_1010/BiasAdd/ReadVariableOp!^dense_1010/MatMul/ReadVariableOp"^dense_1011/BiasAdd/ReadVariableOp!^dense_1011/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¸
_input_shapes¦
£:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization_904/AssignMovingAvg'batch_normalization_904/AssignMovingAvg2p
6batch_normalization_904/AssignMovingAvg/ReadVariableOp6batch_normalization_904/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_904/AssignMovingAvg_1)batch_normalization_904/AssignMovingAvg_12t
8batch_normalization_904/AssignMovingAvg_1/ReadVariableOp8batch_normalization_904/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_904/batchnorm/ReadVariableOp0batch_normalization_904/batchnorm/ReadVariableOp2l
4batch_normalization_904/batchnorm/mul/ReadVariableOp4batch_normalization_904/batchnorm/mul/ReadVariableOp2R
'batch_normalization_905/AssignMovingAvg'batch_normalization_905/AssignMovingAvg2p
6batch_normalization_905/AssignMovingAvg/ReadVariableOp6batch_normalization_905/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_905/AssignMovingAvg_1)batch_normalization_905/AssignMovingAvg_12t
8batch_normalization_905/AssignMovingAvg_1/ReadVariableOp8batch_normalization_905/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_905/batchnorm/ReadVariableOp0batch_normalization_905/batchnorm/ReadVariableOp2l
4batch_normalization_905/batchnorm/mul/ReadVariableOp4batch_normalization_905/batchnorm/mul/ReadVariableOp2R
'batch_normalization_906/AssignMovingAvg'batch_normalization_906/AssignMovingAvg2p
6batch_normalization_906/AssignMovingAvg/ReadVariableOp6batch_normalization_906/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_906/AssignMovingAvg_1)batch_normalization_906/AssignMovingAvg_12t
8batch_normalization_906/AssignMovingAvg_1/ReadVariableOp8batch_normalization_906/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_906/batchnorm/ReadVariableOp0batch_normalization_906/batchnorm/ReadVariableOp2l
4batch_normalization_906/batchnorm/mul/ReadVariableOp4batch_normalization_906/batchnorm/mul/ReadVariableOp2R
'batch_normalization_907/AssignMovingAvg'batch_normalization_907/AssignMovingAvg2p
6batch_normalization_907/AssignMovingAvg/ReadVariableOp6batch_normalization_907/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_907/AssignMovingAvg_1)batch_normalization_907/AssignMovingAvg_12t
8batch_normalization_907/AssignMovingAvg_1/ReadVariableOp8batch_normalization_907/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_907/batchnorm/ReadVariableOp0batch_normalization_907/batchnorm/ReadVariableOp2l
4batch_normalization_907/batchnorm/mul/ReadVariableOp4batch_normalization_907/batchnorm/mul/ReadVariableOp2R
'batch_normalization_908/AssignMovingAvg'batch_normalization_908/AssignMovingAvg2p
6batch_normalization_908/AssignMovingAvg/ReadVariableOp6batch_normalization_908/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_908/AssignMovingAvg_1)batch_normalization_908/AssignMovingAvg_12t
8batch_normalization_908/AssignMovingAvg_1/ReadVariableOp8batch_normalization_908/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_908/batchnorm/ReadVariableOp0batch_normalization_908/batchnorm/ReadVariableOp2l
4batch_normalization_908/batchnorm/mul/ReadVariableOp4batch_normalization_908/batchnorm/mul/ReadVariableOp2R
'batch_normalization_909/AssignMovingAvg'batch_normalization_909/AssignMovingAvg2p
6batch_normalization_909/AssignMovingAvg/ReadVariableOp6batch_normalization_909/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_909/AssignMovingAvg_1)batch_normalization_909/AssignMovingAvg_12t
8batch_normalization_909/AssignMovingAvg_1/ReadVariableOp8batch_normalization_909/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_909/batchnorm/ReadVariableOp0batch_normalization_909/batchnorm/ReadVariableOp2l
4batch_normalization_909/batchnorm/mul/ReadVariableOp4batch_normalization_909/batchnorm/mul/ReadVariableOp2R
'batch_normalization_910/AssignMovingAvg'batch_normalization_910/AssignMovingAvg2p
6batch_normalization_910/AssignMovingAvg/ReadVariableOp6batch_normalization_910/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_910/AssignMovingAvg_1)batch_normalization_910/AssignMovingAvg_12t
8batch_normalization_910/AssignMovingAvg_1/ReadVariableOp8batch_normalization_910/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_910/batchnorm/ReadVariableOp0batch_normalization_910/batchnorm/ReadVariableOp2l
4batch_normalization_910/batchnorm/mul/ReadVariableOp4batch_normalization_910/batchnorm/mul/ReadVariableOp2R
'batch_normalization_911/AssignMovingAvg'batch_normalization_911/AssignMovingAvg2p
6batch_normalization_911/AssignMovingAvg/ReadVariableOp6batch_normalization_911/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_911/AssignMovingAvg_1)batch_normalization_911/AssignMovingAvg_12t
8batch_normalization_911/AssignMovingAvg_1/ReadVariableOp8batch_normalization_911/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_911/batchnorm/ReadVariableOp0batch_normalization_911/batchnorm/ReadVariableOp2l
4batch_normalization_911/batchnorm/mul/ReadVariableOp4batch_normalization_911/batchnorm/mul/ReadVariableOp2R
'batch_normalization_912/AssignMovingAvg'batch_normalization_912/AssignMovingAvg2p
6batch_normalization_912/AssignMovingAvg/ReadVariableOp6batch_normalization_912/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_912/AssignMovingAvg_1)batch_normalization_912/AssignMovingAvg_12t
8batch_normalization_912/AssignMovingAvg_1/ReadVariableOp8batch_normalization_912/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_912/batchnorm/ReadVariableOp0batch_normalization_912/batchnorm/ReadVariableOp2l
4batch_normalization_912/batchnorm/mul/ReadVariableOp4batch_normalization_912/batchnorm/mul/ReadVariableOp2R
'batch_normalization_913/AssignMovingAvg'batch_normalization_913/AssignMovingAvg2p
6batch_normalization_913/AssignMovingAvg/ReadVariableOp6batch_normalization_913/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_913/AssignMovingAvg_1)batch_normalization_913/AssignMovingAvg_12t
8batch_normalization_913/AssignMovingAvg_1/ReadVariableOp8batch_normalization_913/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_913/batchnorm/ReadVariableOp0batch_normalization_913/batchnorm/ReadVariableOp2l
4batch_normalization_913/batchnorm/mul/ReadVariableOp4batch_normalization_913/batchnorm/mul/ReadVariableOp2F
!dense_1001/BiasAdd/ReadVariableOp!dense_1001/BiasAdd/ReadVariableOp2D
 dense_1001/MatMul/ReadVariableOp dense_1001/MatMul/ReadVariableOp2F
!dense_1002/BiasAdd/ReadVariableOp!dense_1002/BiasAdd/ReadVariableOp2D
 dense_1002/MatMul/ReadVariableOp dense_1002/MatMul/ReadVariableOp2F
!dense_1003/BiasAdd/ReadVariableOp!dense_1003/BiasAdd/ReadVariableOp2D
 dense_1003/MatMul/ReadVariableOp dense_1003/MatMul/ReadVariableOp2F
!dense_1004/BiasAdd/ReadVariableOp!dense_1004/BiasAdd/ReadVariableOp2D
 dense_1004/MatMul/ReadVariableOp dense_1004/MatMul/ReadVariableOp2F
!dense_1005/BiasAdd/ReadVariableOp!dense_1005/BiasAdd/ReadVariableOp2D
 dense_1005/MatMul/ReadVariableOp dense_1005/MatMul/ReadVariableOp2F
!dense_1006/BiasAdd/ReadVariableOp!dense_1006/BiasAdd/ReadVariableOp2D
 dense_1006/MatMul/ReadVariableOp dense_1006/MatMul/ReadVariableOp2F
!dense_1007/BiasAdd/ReadVariableOp!dense_1007/BiasAdd/ReadVariableOp2D
 dense_1007/MatMul/ReadVariableOp dense_1007/MatMul/ReadVariableOp2F
!dense_1008/BiasAdd/ReadVariableOp!dense_1008/BiasAdd/ReadVariableOp2D
 dense_1008/MatMul/ReadVariableOp dense_1008/MatMul/ReadVariableOp2F
!dense_1009/BiasAdd/ReadVariableOp!dense_1009/BiasAdd/ReadVariableOp2D
 dense_1009/MatMul/ReadVariableOp dense_1009/MatMul/ReadVariableOp2F
!dense_1010/BiasAdd/ReadVariableOp!dense_1010/BiasAdd/ReadVariableOp2D
 dense_1010/MatMul/ReadVariableOp dense_1010/MatMul/ReadVariableOp2F
!dense_1011/BiasAdd/ReadVariableOp!dense_1011/BiasAdd/ReadVariableOp2D
 dense_1011/MatMul/ReadVariableOp dense_1011/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
«
Ô
9__inference_batch_normalization_908_layer_call_fn_1003380

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_908_layer_call_and_return_conditional_losses_999843o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_911_layer_call_and_return_conditional_losses_1003727

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
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
S__inference_batch_normalization_908_layer_call_and_return_conditional_losses_999796

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1006_layer_call_and_return_conditional_losses_1000448

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_912_layer_call_fn_1003816

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_912_layer_call_and_return_conditional_losses_1000171o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1003_layer_call_and_return_conditional_losses_1003136

inputs0
matmul_readvariableop_resource:5-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
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
L__inference_leaky_re_lu_909_layer_call_and_return_conditional_losses_1000468

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
öå
Ì9
J__inference_sequential_97_layer_call_and_return_conditional_losses_1002330

inputs
normalization_97_sub_y
normalization_97_sqrt_x;
)dense_1001_matmul_readvariableop_resource:58
*dense_1001_biasadd_readvariableop_resource:5G
9batch_normalization_904_batchnorm_readvariableop_resource:5K
=batch_normalization_904_batchnorm_mul_readvariableop_resource:5I
;batch_normalization_904_batchnorm_readvariableop_1_resource:5I
;batch_normalization_904_batchnorm_readvariableop_2_resource:5;
)dense_1002_matmul_readvariableop_resource:558
*dense_1002_biasadd_readvariableop_resource:5G
9batch_normalization_905_batchnorm_readvariableop_resource:5K
=batch_normalization_905_batchnorm_mul_readvariableop_resource:5I
;batch_normalization_905_batchnorm_readvariableop_1_resource:5I
;batch_normalization_905_batchnorm_readvariableop_2_resource:5;
)dense_1003_matmul_readvariableop_resource:58
*dense_1003_biasadd_readvariableop_resource:G
9batch_normalization_906_batchnorm_readvariableop_resource:K
=batch_normalization_906_batchnorm_mul_readvariableop_resource:I
;batch_normalization_906_batchnorm_readvariableop_1_resource:I
;batch_normalization_906_batchnorm_readvariableop_2_resource:;
)dense_1004_matmul_readvariableop_resource:8
*dense_1004_biasadd_readvariableop_resource:G
9batch_normalization_907_batchnorm_readvariableop_resource:K
=batch_normalization_907_batchnorm_mul_readvariableop_resource:I
;batch_normalization_907_batchnorm_readvariableop_1_resource:I
;batch_normalization_907_batchnorm_readvariableop_2_resource:;
)dense_1005_matmul_readvariableop_resource:8
*dense_1005_biasadd_readvariableop_resource:G
9batch_normalization_908_batchnorm_readvariableop_resource:K
=batch_normalization_908_batchnorm_mul_readvariableop_resource:I
;batch_normalization_908_batchnorm_readvariableop_1_resource:I
;batch_normalization_908_batchnorm_readvariableop_2_resource:;
)dense_1006_matmul_readvariableop_resource:8
*dense_1006_biasadd_readvariableop_resource:G
9batch_normalization_909_batchnorm_readvariableop_resource:K
=batch_normalization_909_batchnorm_mul_readvariableop_resource:I
;batch_normalization_909_batchnorm_readvariableop_1_resource:I
;batch_normalization_909_batchnorm_readvariableop_2_resource:;
)dense_1007_matmul_readvariableop_resource:8
*dense_1007_biasadd_readvariableop_resource:G
9batch_normalization_910_batchnorm_readvariableop_resource:K
=batch_normalization_910_batchnorm_mul_readvariableop_resource:I
;batch_normalization_910_batchnorm_readvariableop_1_resource:I
;batch_normalization_910_batchnorm_readvariableop_2_resource:;
)dense_1008_matmul_readvariableop_resource:8
*dense_1008_biasadd_readvariableop_resource:G
9batch_normalization_911_batchnorm_readvariableop_resource:K
=batch_normalization_911_batchnorm_mul_readvariableop_resource:I
;batch_normalization_911_batchnorm_readvariableop_1_resource:I
;batch_normalization_911_batchnorm_readvariableop_2_resource:;
)dense_1009_matmul_readvariableop_resource:8
*dense_1009_biasadd_readvariableop_resource:G
9batch_normalization_912_batchnorm_readvariableop_resource:K
=batch_normalization_912_batchnorm_mul_readvariableop_resource:I
;batch_normalization_912_batchnorm_readvariableop_1_resource:I
;batch_normalization_912_batchnorm_readvariableop_2_resource:;
)dense_1010_matmul_readvariableop_resource:8
*dense_1010_biasadd_readvariableop_resource:G
9batch_normalization_913_batchnorm_readvariableop_resource:K
=batch_normalization_913_batchnorm_mul_readvariableop_resource:I
;batch_normalization_913_batchnorm_readvariableop_1_resource:I
;batch_normalization_913_batchnorm_readvariableop_2_resource:;
)dense_1011_matmul_readvariableop_resource:8
*dense_1011_biasadd_readvariableop_resource:
identity¢0batch_normalization_904/batchnorm/ReadVariableOp¢2batch_normalization_904/batchnorm/ReadVariableOp_1¢2batch_normalization_904/batchnorm/ReadVariableOp_2¢4batch_normalization_904/batchnorm/mul/ReadVariableOp¢0batch_normalization_905/batchnorm/ReadVariableOp¢2batch_normalization_905/batchnorm/ReadVariableOp_1¢2batch_normalization_905/batchnorm/ReadVariableOp_2¢4batch_normalization_905/batchnorm/mul/ReadVariableOp¢0batch_normalization_906/batchnorm/ReadVariableOp¢2batch_normalization_906/batchnorm/ReadVariableOp_1¢2batch_normalization_906/batchnorm/ReadVariableOp_2¢4batch_normalization_906/batchnorm/mul/ReadVariableOp¢0batch_normalization_907/batchnorm/ReadVariableOp¢2batch_normalization_907/batchnorm/ReadVariableOp_1¢2batch_normalization_907/batchnorm/ReadVariableOp_2¢4batch_normalization_907/batchnorm/mul/ReadVariableOp¢0batch_normalization_908/batchnorm/ReadVariableOp¢2batch_normalization_908/batchnorm/ReadVariableOp_1¢2batch_normalization_908/batchnorm/ReadVariableOp_2¢4batch_normalization_908/batchnorm/mul/ReadVariableOp¢0batch_normalization_909/batchnorm/ReadVariableOp¢2batch_normalization_909/batchnorm/ReadVariableOp_1¢2batch_normalization_909/batchnorm/ReadVariableOp_2¢4batch_normalization_909/batchnorm/mul/ReadVariableOp¢0batch_normalization_910/batchnorm/ReadVariableOp¢2batch_normalization_910/batchnorm/ReadVariableOp_1¢2batch_normalization_910/batchnorm/ReadVariableOp_2¢4batch_normalization_910/batchnorm/mul/ReadVariableOp¢0batch_normalization_911/batchnorm/ReadVariableOp¢2batch_normalization_911/batchnorm/ReadVariableOp_1¢2batch_normalization_911/batchnorm/ReadVariableOp_2¢4batch_normalization_911/batchnorm/mul/ReadVariableOp¢0batch_normalization_912/batchnorm/ReadVariableOp¢2batch_normalization_912/batchnorm/ReadVariableOp_1¢2batch_normalization_912/batchnorm/ReadVariableOp_2¢4batch_normalization_912/batchnorm/mul/ReadVariableOp¢0batch_normalization_913/batchnorm/ReadVariableOp¢2batch_normalization_913/batchnorm/ReadVariableOp_1¢2batch_normalization_913/batchnorm/ReadVariableOp_2¢4batch_normalization_913/batchnorm/mul/ReadVariableOp¢!dense_1001/BiasAdd/ReadVariableOp¢ dense_1001/MatMul/ReadVariableOp¢!dense_1002/BiasAdd/ReadVariableOp¢ dense_1002/MatMul/ReadVariableOp¢!dense_1003/BiasAdd/ReadVariableOp¢ dense_1003/MatMul/ReadVariableOp¢!dense_1004/BiasAdd/ReadVariableOp¢ dense_1004/MatMul/ReadVariableOp¢!dense_1005/BiasAdd/ReadVariableOp¢ dense_1005/MatMul/ReadVariableOp¢!dense_1006/BiasAdd/ReadVariableOp¢ dense_1006/MatMul/ReadVariableOp¢!dense_1007/BiasAdd/ReadVariableOp¢ dense_1007/MatMul/ReadVariableOp¢!dense_1008/BiasAdd/ReadVariableOp¢ dense_1008/MatMul/ReadVariableOp¢!dense_1009/BiasAdd/ReadVariableOp¢ dense_1009/MatMul/ReadVariableOp¢!dense_1010/BiasAdd/ReadVariableOp¢ dense_1010/MatMul/ReadVariableOp¢!dense_1011/BiasAdd/ReadVariableOp¢ dense_1011/MatMul/ReadVariableOpm
normalization_97/subSubinputsnormalization_97_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_97/SqrtSqrtnormalization_97_sqrt_x*
T0*
_output_shapes

:_
normalization_97/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_97/MaximumMaximumnormalization_97/Sqrt:y:0#normalization_97/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_97/truedivRealDivnormalization_97/sub:z:0normalization_97/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_1001/MatMul/ReadVariableOpReadVariableOp)dense_1001_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0
dense_1001/MatMulMatMulnormalization_97/truediv:z:0(dense_1001/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
!dense_1001/BiasAdd/ReadVariableOpReadVariableOp*dense_1001_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_1001/BiasAddBiasAdddense_1001/MatMul:product:0)dense_1001/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¦
0batch_normalization_904/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_904_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0l
'batch_normalization_904/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_904/batchnorm/addAddV28batch_normalization_904/batchnorm/ReadVariableOp:value:00batch_normalization_904/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
'batch_normalization_904/batchnorm/RsqrtRsqrt)batch_normalization_904/batchnorm/add:z:0*
T0*
_output_shapes
:5®
4batch_normalization_904/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_904_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0¼
%batch_normalization_904/batchnorm/mulMul+batch_normalization_904/batchnorm/Rsqrt:y:0<batch_normalization_904/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5¨
'batch_normalization_904/batchnorm/mul_1Muldense_1001/BiasAdd:output:0)batch_normalization_904/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ª
2batch_normalization_904/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_904_batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0º
'batch_normalization_904/batchnorm/mul_2Mul:batch_normalization_904/batchnorm/ReadVariableOp_1:value:0)batch_normalization_904/batchnorm/mul:z:0*
T0*
_output_shapes
:5ª
2batch_normalization_904/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_904_batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0º
%batch_normalization_904/batchnorm/subSub:batch_normalization_904/batchnorm/ReadVariableOp_2:value:0+batch_normalization_904/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5º
'batch_normalization_904/batchnorm/add_1AddV2+batch_normalization_904/batchnorm/mul_1:z:0)batch_normalization_904/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
leaky_re_lu_904/LeakyRelu	LeakyRelu+batch_normalization_904/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>
 dense_1002/MatMul/ReadVariableOpReadVariableOp)dense_1002_matmul_readvariableop_resource*
_output_shapes

:55*
dtype0 
dense_1002/MatMulMatMul'leaky_re_lu_904/LeakyRelu:activations:0(dense_1002/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
!dense_1002/BiasAdd/ReadVariableOpReadVariableOp*dense_1002_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_1002/BiasAddBiasAdddense_1002/MatMul:product:0)dense_1002/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5¦
0batch_normalization_905/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_905_batchnorm_readvariableop_resource*
_output_shapes
:5*
dtype0l
'batch_normalization_905/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_905/batchnorm/addAddV28batch_normalization_905/batchnorm/ReadVariableOp:value:00batch_normalization_905/batchnorm/add/y:output:0*
T0*
_output_shapes
:5
'batch_normalization_905/batchnorm/RsqrtRsqrt)batch_normalization_905/batchnorm/add:z:0*
T0*
_output_shapes
:5®
4batch_normalization_905/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_905_batchnorm_mul_readvariableop_resource*
_output_shapes
:5*
dtype0¼
%batch_normalization_905/batchnorm/mulMul+batch_normalization_905/batchnorm/Rsqrt:y:0<batch_normalization_905/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:5¨
'batch_normalization_905/batchnorm/mul_1Muldense_1002/BiasAdd:output:0)batch_normalization_905/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5ª
2batch_normalization_905/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_905_batchnorm_readvariableop_1_resource*
_output_shapes
:5*
dtype0º
'batch_normalization_905/batchnorm/mul_2Mul:batch_normalization_905/batchnorm/ReadVariableOp_1:value:0)batch_normalization_905/batchnorm/mul:z:0*
T0*
_output_shapes
:5ª
2batch_normalization_905/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_905_batchnorm_readvariableop_2_resource*
_output_shapes
:5*
dtype0º
%batch_normalization_905/batchnorm/subSub:batch_normalization_905/batchnorm/ReadVariableOp_2:value:0+batch_normalization_905/batchnorm/mul_2:z:0*
T0*
_output_shapes
:5º
'batch_normalization_905/batchnorm/add_1AddV2+batch_normalization_905/batchnorm/mul_1:z:0)batch_normalization_905/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
leaky_re_lu_905/LeakyRelu	LeakyRelu+batch_normalization_905/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5*
alpha%>
 dense_1003/MatMul/ReadVariableOpReadVariableOp)dense_1003_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0 
dense_1003/MatMulMatMul'leaky_re_lu_905/LeakyRelu:activations:0(dense_1003/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1003/BiasAdd/ReadVariableOpReadVariableOp*dense_1003_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1003/BiasAddBiasAdddense_1003/MatMul:product:0)dense_1003/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_906/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_906_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_906/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_906/batchnorm/addAddV28batch_normalization_906/batchnorm/ReadVariableOp:value:00batch_normalization_906/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_906/batchnorm/RsqrtRsqrt)batch_normalization_906/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_906/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_906_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_906/batchnorm/mulMul+batch_normalization_906/batchnorm/Rsqrt:y:0<batch_normalization_906/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_906/batchnorm/mul_1Muldense_1003/BiasAdd:output:0)batch_normalization_906/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_906/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_906_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_906/batchnorm/mul_2Mul:batch_normalization_906/batchnorm/ReadVariableOp_1:value:0)batch_normalization_906/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_906/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_906_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_906/batchnorm/subSub:batch_normalization_906/batchnorm/ReadVariableOp_2:value:0+batch_normalization_906/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_906/batchnorm/add_1AddV2+batch_normalization_906/batchnorm/mul_1:z:0)batch_normalization_906/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_906/LeakyRelu	LeakyRelu+batch_normalization_906/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1004/MatMul/ReadVariableOpReadVariableOp)dense_1004_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1004/MatMulMatMul'leaky_re_lu_906/LeakyRelu:activations:0(dense_1004/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1004/BiasAdd/ReadVariableOpReadVariableOp*dense_1004_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1004/BiasAddBiasAdddense_1004/MatMul:product:0)dense_1004/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_907/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_907_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_907/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_907/batchnorm/addAddV28batch_normalization_907/batchnorm/ReadVariableOp:value:00batch_normalization_907/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_907/batchnorm/RsqrtRsqrt)batch_normalization_907/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_907/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_907_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_907/batchnorm/mulMul+batch_normalization_907/batchnorm/Rsqrt:y:0<batch_normalization_907/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_907/batchnorm/mul_1Muldense_1004/BiasAdd:output:0)batch_normalization_907/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_907/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_907_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_907/batchnorm/mul_2Mul:batch_normalization_907/batchnorm/ReadVariableOp_1:value:0)batch_normalization_907/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_907/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_907_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_907/batchnorm/subSub:batch_normalization_907/batchnorm/ReadVariableOp_2:value:0+batch_normalization_907/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_907/batchnorm/add_1AddV2+batch_normalization_907/batchnorm/mul_1:z:0)batch_normalization_907/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_907/LeakyRelu	LeakyRelu+batch_normalization_907/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1005/MatMul/ReadVariableOpReadVariableOp)dense_1005_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1005/MatMulMatMul'leaky_re_lu_907/LeakyRelu:activations:0(dense_1005/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1005/BiasAdd/ReadVariableOpReadVariableOp*dense_1005_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1005/BiasAddBiasAdddense_1005/MatMul:product:0)dense_1005/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_908/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_908_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_908/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_908/batchnorm/addAddV28batch_normalization_908/batchnorm/ReadVariableOp:value:00batch_normalization_908/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_908/batchnorm/RsqrtRsqrt)batch_normalization_908/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_908/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_908_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_908/batchnorm/mulMul+batch_normalization_908/batchnorm/Rsqrt:y:0<batch_normalization_908/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_908/batchnorm/mul_1Muldense_1005/BiasAdd:output:0)batch_normalization_908/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_908/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_908_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_908/batchnorm/mul_2Mul:batch_normalization_908/batchnorm/ReadVariableOp_1:value:0)batch_normalization_908/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_908/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_908_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_908/batchnorm/subSub:batch_normalization_908/batchnorm/ReadVariableOp_2:value:0+batch_normalization_908/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_908/batchnorm/add_1AddV2+batch_normalization_908/batchnorm/mul_1:z:0)batch_normalization_908/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_908/LeakyRelu	LeakyRelu+batch_normalization_908/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1006/MatMul/ReadVariableOpReadVariableOp)dense_1006_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1006/MatMulMatMul'leaky_re_lu_908/LeakyRelu:activations:0(dense_1006/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1006/BiasAdd/ReadVariableOpReadVariableOp*dense_1006_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1006/BiasAddBiasAdddense_1006/MatMul:product:0)dense_1006/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_909/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_909_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_909/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_909/batchnorm/addAddV28batch_normalization_909/batchnorm/ReadVariableOp:value:00batch_normalization_909/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_909/batchnorm/RsqrtRsqrt)batch_normalization_909/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_909/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_909_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_909/batchnorm/mulMul+batch_normalization_909/batchnorm/Rsqrt:y:0<batch_normalization_909/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_909/batchnorm/mul_1Muldense_1006/BiasAdd:output:0)batch_normalization_909/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_909/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_909_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_909/batchnorm/mul_2Mul:batch_normalization_909/batchnorm/ReadVariableOp_1:value:0)batch_normalization_909/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_909/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_909_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_909/batchnorm/subSub:batch_normalization_909/batchnorm/ReadVariableOp_2:value:0+batch_normalization_909/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_909/batchnorm/add_1AddV2+batch_normalization_909/batchnorm/mul_1:z:0)batch_normalization_909/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_909/LeakyRelu	LeakyRelu+batch_normalization_909/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1007/MatMul/ReadVariableOpReadVariableOp)dense_1007_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1007/MatMulMatMul'leaky_re_lu_909/LeakyRelu:activations:0(dense_1007/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1007/BiasAdd/ReadVariableOpReadVariableOp*dense_1007_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1007/BiasAddBiasAdddense_1007/MatMul:product:0)dense_1007/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_910/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_910_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_910/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_910/batchnorm/addAddV28batch_normalization_910/batchnorm/ReadVariableOp:value:00batch_normalization_910/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_910/batchnorm/RsqrtRsqrt)batch_normalization_910/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_910/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_910_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_910/batchnorm/mulMul+batch_normalization_910/batchnorm/Rsqrt:y:0<batch_normalization_910/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_910/batchnorm/mul_1Muldense_1007/BiasAdd:output:0)batch_normalization_910/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_910/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_910_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_910/batchnorm/mul_2Mul:batch_normalization_910/batchnorm/ReadVariableOp_1:value:0)batch_normalization_910/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_910/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_910_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_910/batchnorm/subSub:batch_normalization_910/batchnorm/ReadVariableOp_2:value:0+batch_normalization_910/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_910/batchnorm/add_1AddV2+batch_normalization_910/batchnorm/mul_1:z:0)batch_normalization_910/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_910/LeakyRelu	LeakyRelu+batch_normalization_910/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1008/MatMul/ReadVariableOpReadVariableOp)dense_1008_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1008/MatMulMatMul'leaky_re_lu_910/LeakyRelu:activations:0(dense_1008/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1008/BiasAdd/ReadVariableOpReadVariableOp*dense_1008_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1008/BiasAddBiasAdddense_1008/MatMul:product:0)dense_1008/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_911/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_911_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_911/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_911/batchnorm/addAddV28batch_normalization_911/batchnorm/ReadVariableOp:value:00batch_normalization_911/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_911/batchnorm/RsqrtRsqrt)batch_normalization_911/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_911/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_911_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_911/batchnorm/mulMul+batch_normalization_911/batchnorm/Rsqrt:y:0<batch_normalization_911/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_911/batchnorm/mul_1Muldense_1008/BiasAdd:output:0)batch_normalization_911/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_911/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_911_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_911/batchnorm/mul_2Mul:batch_normalization_911/batchnorm/ReadVariableOp_1:value:0)batch_normalization_911/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_911/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_911_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_911/batchnorm/subSub:batch_normalization_911/batchnorm/ReadVariableOp_2:value:0+batch_normalization_911/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_911/batchnorm/add_1AddV2+batch_normalization_911/batchnorm/mul_1:z:0)batch_normalization_911/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_911/LeakyRelu	LeakyRelu+batch_normalization_911/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1009/MatMul/ReadVariableOpReadVariableOp)dense_1009_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1009/MatMulMatMul'leaky_re_lu_911/LeakyRelu:activations:0(dense_1009/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1009/BiasAdd/ReadVariableOpReadVariableOp*dense_1009_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1009/BiasAddBiasAdddense_1009/MatMul:product:0)dense_1009/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_912/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_912_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_912/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_912/batchnorm/addAddV28batch_normalization_912/batchnorm/ReadVariableOp:value:00batch_normalization_912/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_912/batchnorm/RsqrtRsqrt)batch_normalization_912/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_912/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_912_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_912/batchnorm/mulMul+batch_normalization_912/batchnorm/Rsqrt:y:0<batch_normalization_912/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_912/batchnorm/mul_1Muldense_1009/BiasAdd:output:0)batch_normalization_912/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_912/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_912_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_912/batchnorm/mul_2Mul:batch_normalization_912/batchnorm/ReadVariableOp_1:value:0)batch_normalization_912/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_912/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_912_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_912/batchnorm/subSub:batch_normalization_912/batchnorm/ReadVariableOp_2:value:0+batch_normalization_912/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_912/batchnorm/add_1AddV2+batch_normalization_912/batchnorm/mul_1:z:0)batch_normalization_912/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_912/LeakyRelu	LeakyRelu+batch_normalization_912/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1010/MatMul/ReadVariableOpReadVariableOp)dense_1010_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1010/MatMulMatMul'leaky_re_lu_912/LeakyRelu:activations:0(dense_1010/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1010/BiasAdd/ReadVariableOpReadVariableOp*dense_1010_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1010/BiasAddBiasAdddense_1010/MatMul:product:0)dense_1010/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_913/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_913_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_913/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_913/batchnorm/addAddV28batch_normalization_913/batchnorm/ReadVariableOp:value:00batch_normalization_913/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_913/batchnorm/RsqrtRsqrt)batch_normalization_913/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_913/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_913_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_913/batchnorm/mulMul+batch_normalization_913/batchnorm/Rsqrt:y:0<batch_normalization_913/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¨
'batch_normalization_913/batchnorm/mul_1Muldense_1010/BiasAdd:output:0)batch_normalization_913/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_913/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_913_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_913/batchnorm/mul_2Mul:batch_normalization_913/batchnorm/ReadVariableOp_1:value:0)batch_normalization_913/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_913/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_913_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_913/batchnorm/subSub:batch_normalization_913/batchnorm/ReadVariableOp_2:value:0+batch_normalization_913/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_913/batchnorm/add_1AddV2+batch_normalization_913/batchnorm/mul_1:z:0)batch_normalization_913/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_913/LeakyRelu	LeakyRelu+batch_normalization_913/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
 dense_1011/MatMul/ReadVariableOpReadVariableOp)dense_1011_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
dense_1011/MatMulMatMul'leaky_re_lu_913/LeakyRelu:activations:0(dense_1011/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_1011/BiasAdd/ReadVariableOpReadVariableOp*dense_1011_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1011/BiasAddBiasAdddense_1011/MatMul:product:0)dense_1011/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitydense_1011/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp1^batch_normalization_904/batchnorm/ReadVariableOp3^batch_normalization_904/batchnorm/ReadVariableOp_13^batch_normalization_904/batchnorm/ReadVariableOp_25^batch_normalization_904/batchnorm/mul/ReadVariableOp1^batch_normalization_905/batchnorm/ReadVariableOp3^batch_normalization_905/batchnorm/ReadVariableOp_13^batch_normalization_905/batchnorm/ReadVariableOp_25^batch_normalization_905/batchnorm/mul/ReadVariableOp1^batch_normalization_906/batchnorm/ReadVariableOp3^batch_normalization_906/batchnorm/ReadVariableOp_13^batch_normalization_906/batchnorm/ReadVariableOp_25^batch_normalization_906/batchnorm/mul/ReadVariableOp1^batch_normalization_907/batchnorm/ReadVariableOp3^batch_normalization_907/batchnorm/ReadVariableOp_13^batch_normalization_907/batchnorm/ReadVariableOp_25^batch_normalization_907/batchnorm/mul/ReadVariableOp1^batch_normalization_908/batchnorm/ReadVariableOp3^batch_normalization_908/batchnorm/ReadVariableOp_13^batch_normalization_908/batchnorm/ReadVariableOp_25^batch_normalization_908/batchnorm/mul/ReadVariableOp1^batch_normalization_909/batchnorm/ReadVariableOp3^batch_normalization_909/batchnorm/ReadVariableOp_13^batch_normalization_909/batchnorm/ReadVariableOp_25^batch_normalization_909/batchnorm/mul/ReadVariableOp1^batch_normalization_910/batchnorm/ReadVariableOp3^batch_normalization_910/batchnorm/ReadVariableOp_13^batch_normalization_910/batchnorm/ReadVariableOp_25^batch_normalization_910/batchnorm/mul/ReadVariableOp1^batch_normalization_911/batchnorm/ReadVariableOp3^batch_normalization_911/batchnorm/ReadVariableOp_13^batch_normalization_911/batchnorm/ReadVariableOp_25^batch_normalization_911/batchnorm/mul/ReadVariableOp1^batch_normalization_912/batchnorm/ReadVariableOp3^batch_normalization_912/batchnorm/ReadVariableOp_13^batch_normalization_912/batchnorm/ReadVariableOp_25^batch_normalization_912/batchnorm/mul/ReadVariableOp1^batch_normalization_913/batchnorm/ReadVariableOp3^batch_normalization_913/batchnorm/ReadVariableOp_13^batch_normalization_913/batchnorm/ReadVariableOp_25^batch_normalization_913/batchnorm/mul/ReadVariableOp"^dense_1001/BiasAdd/ReadVariableOp!^dense_1001/MatMul/ReadVariableOp"^dense_1002/BiasAdd/ReadVariableOp!^dense_1002/MatMul/ReadVariableOp"^dense_1003/BiasAdd/ReadVariableOp!^dense_1003/MatMul/ReadVariableOp"^dense_1004/BiasAdd/ReadVariableOp!^dense_1004/MatMul/ReadVariableOp"^dense_1005/BiasAdd/ReadVariableOp!^dense_1005/MatMul/ReadVariableOp"^dense_1006/BiasAdd/ReadVariableOp!^dense_1006/MatMul/ReadVariableOp"^dense_1007/BiasAdd/ReadVariableOp!^dense_1007/MatMul/ReadVariableOp"^dense_1008/BiasAdd/ReadVariableOp!^dense_1008/MatMul/ReadVariableOp"^dense_1009/BiasAdd/ReadVariableOp!^dense_1009/MatMul/ReadVariableOp"^dense_1010/BiasAdd/ReadVariableOp!^dense_1010/MatMul/ReadVariableOp"^dense_1011/BiasAdd/ReadVariableOp!^dense_1011/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¸
_input_shapes¦
£:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0batch_normalization_904/batchnorm/ReadVariableOp0batch_normalization_904/batchnorm/ReadVariableOp2h
2batch_normalization_904/batchnorm/ReadVariableOp_12batch_normalization_904/batchnorm/ReadVariableOp_12h
2batch_normalization_904/batchnorm/ReadVariableOp_22batch_normalization_904/batchnorm/ReadVariableOp_22l
4batch_normalization_904/batchnorm/mul/ReadVariableOp4batch_normalization_904/batchnorm/mul/ReadVariableOp2d
0batch_normalization_905/batchnorm/ReadVariableOp0batch_normalization_905/batchnorm/ReadVariableOp2h
2batch_normalization_905/batchnorm/ReadVariableOp_12batch_normalization_905/batchnorm/ReadVariableOp_12h
2batch_normalization_905/batchnorm/ReadVariableOp_22batch_normalization_905/batchnorm/ReadVariableOp_22l
4batch_normalization_905/batchnorm/mul/ReadVariableOp4batch_normalization_905/batchnorm/mul/ReadVariableOp2d
0batch_normalization_906/batchnorm/ReadVariableOp0batch_normalization_906/batchnorm/ReadVariableOp2h
2batch_normalization_906/batchnorm/ReadVariableOp_12batch_normalization_906/batchnorm/ReadVariableOp_12h
2batch_normalization_906/batchnorm/ReadVariableOp_22batch_normalization_906/batchnorm/ReadVariableOp_22l
4batch_normalization_906/batchnorm/mul/ReadVariableOp4batch_normalization_906/batchnorm/mul/ReadVariableOp2d
0batch_normalization_907/batchnorm/ReadVariableOp0batch_normalization_907/batchnorm/ReadVariableOp2h
2batch_normalization_907/batchnorm/ReadVariableOp_12batch_normalization_907/batchnorm/ReadVariableOp_12h
2batch_normalization_907/batchnorm/ReadVariableOp_22batch_normalization_907/batchnorm/ReadVariableOp_22l
4batch_normalization_907/batchnorm/mul/ReadVariableOp4batch_normalization_907/batchnorm/mul/ReadVariableOp2d
0batch_normalization_908/batchnorm/ReadVariableOp0batch_normalization_908/batchnorm/ReadVariableOp2h
2batch_normalization_908/batchnorm/ReadVariableOp_12batch_normalization_908/batchnorm/ReadVariableOp_12h
2batch_normalization_908/batchnorm/ReadVariableOp_22batch_normalization_908/batchnorm/ReadVariableOp_22l
4batch_normalization_908/batchnorm/mul/ReadVariableOp4batch_normalization_908/batchnorm/mul/ReadVariableOp2d
0batch_normalization_909/batchnorm/ReadVariableOp0batch_normalization_909/batchnorm/ReadVariableOp2h
2batch_normalization_909/batchnorm/ReadVariableOp_12batch_normalization_909/batchnorm/ReadVariableOp_12h
2batch_normalization_909/batchnorm/ReadVariableOp_22batch_normalization_909/batchnorm/ReadVariableOp_22l
4batch_normalization_909/batchnorm/mul/ReadVariableOp4batch_normalization_909/batchnorm/mul/ReadVariableOp2d
0batch_normalization_910/batchnorm/ReadVariableOp0batch_normalization_910/batchnorm/ReadVariableOp2h
2batch_normalization_910/batchnorm/ReadVariableOp_12batch_normalization_910/batchnorm/ReadVariableOp_12h
2batch_normalization_910/batchnorm/ReadVariableOp_22batch_normalization_910/batchnorm/ReadVariableOp_22l
4batch_normalization_910/batchnorm/mul/ReadVariableOp4batch_normalization_910/batchnorm/mul/ReadVariableOp2d
0batch_normalization_911/batchnorm/ReadVariableOp0batch_normalization_911/batchnorm/ReadVariableOp2h
2batch_normalization_911/batchnorm/ReadVariableOp_12batch_normalization_911/batchnorm/ReadVariableOp_12h
2batch_normalization_911/batchnorm/ReadVariableOp_22batch_normalization_911/batchnorm/ReadVariableOp_22l
4batch_normalization_911/batchnorm/mul/ReadVariableOp4batch_normalization_911/batchnorm/mul/ReadVariableOp2d
0batch_normalization_912/batchnorm/ReadVariableOp0batch_normalization_912/batchnorm/ReadVariableOp2h
2batch_normalization_912/batchnorm/ReadVariableOp_12batch_normalization_912/batchnorm/ReadVariableOp_12h
2batch_normalization_912/batchnorm/ReadVariableOp_22batch_normalization_912/batchnorm/ReadVariableOp_22l
4batch_normalization_912/batchnorm/mul/ReadVariableOp4batch_normalization_912/batchnorm/mul/ReadVariableOp2d
0batch_normalization_913/batchnorm/ReadVariableOp0batch_normalization_913/batchnorm/ReadVariableOp2h
2batch_normalization_913/batchnorm/ReadVariableOp_12batch_normalization_913/batchnorm/ReadVariableOp_12h
2batch_normalization_913/batchnorm/ReadVariableOp_22batch_normalization_913/batchnorm/ReadVariableOp_22l
4batch_normalization_913/batchnorm/mul/ReadVariableOp4batch_normalization_913/batchnorm/mul/ReadVariableOp2F
!dense_1001/BiasAdd/ReadVariableOp!dense_1001/BiasAdd/ReadVariableOp2D
 dense_1001/MatMul/ReadVariableOp dense_1001/MatMul/ReadVariableOp2F
!dense_1002/BiasAdd/ReadVariableOp!dense_1002/BiasAdd/ReadVariableOp2D
 dense_1002/MatMul/ReadVariableOp dense_1002/MatMul/ReadVariableOp2F
!dense_1003/BiasAdd/ReadVariableOp!dense_1003/BiasAdd/ReadVariableOp2D
 dense_1003/MatMul/ReadVariableOp dense_1003/MatMul/ReadVariableOp2F
!dense_1004/BiasAdd/ReadVariableOp!dense_1004/BiasAdd/ReadVariableOp2D
 dense_1004/MatMul/ReadVariableOp dense_1004/MatMul/ReadVariableOp2F
!dense_1005/BiasAdd/ReadVariableOp!dense_1005/BiasAdd/ReadVariableOp2D
 dense_1005/MatMul/ReadVariableOp dense_1005/MatMul/ReadVariableOp2F
!dense_1006/BiasAdd/ReadVariableOp!dense_1006/BiasAdd/ReadVariableOp2D
 dense_1006/MatMul/ReadVariableOp dense_1006/MatMul/ReadVariableOp2F
!dense_1007/BiasAdd/ReadVariableOp!dense_1007/BiasAdd/ReadVariableOp2D
 dense_1007/MatMul/ReadVariableOp dense_1007/MatMul/ReadVariableOp2F
!dense_1008/BiasAdd/ReadVariableOp!dense_1008/BiasAdd/ReadVariableOp2D
 dense_1008/MatMul/ReadVariableOp dense_1008/MatMul/ReadVariableOp2F
!dense_1009/BiasAdd/ReadVariableOp!dense_1009/BiasAdd/ReadVariableOp2D
 dense_1009/MatMul/ReadVariableOp dense_1009/MatMul/ReadVariableOp2F
!dense_1010/BiasAdd/ReadVariableOp!dense_1010/BiasAdd/ReadVariableOp2D
 dense_1010/MatMul/ReadVariableOp dense_1010/MatMul/ReadVariableOp2F
!dense_1011/BiasAdd/ReadVariableOp!dense_1011/BiasAdd/ReadVariableOp2D
 dense_1011/MatMul/ReadVariableOp dense_1011/MatMul/ReadVariableOp:O K
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
T__inference_batch_normalization_906_layer_call_and_return_conditional_losses_1003216

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_906_layer_call_and_return_conditional_losses_1003226

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1008_layer_call_and_return_conditional_losses_1000512

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_905_layer_call_and_return_conditional_losses_999550

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
T__inference_batch_normalization_910_layer_call_and_return_conditional_losses_1003652

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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
æ
h
L__inference_leaky_re_lu_912_layer_call_and_return_conditional_losses_1003880

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_910_layer_call_fn_1003598

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_910_layer_call_and_return_conditional_losses_1000007o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_906_layer_call_and_return_conditional_losses_1000372

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_907_layer_call_and_return_conditional_losses_1003335

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_913_layer_call_fn_1003984

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_913_layer_call_and_return_conditional_losses_1000596`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
ì
S__inference_batch_normalization_907_layer_call_and_return_conditional_losses_999761

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1010_layer_call_and_return_conditional_losses_1003899

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å
ó
/__inference_sequential_97_layer_call_fn_1002083

inputs
unknown
	unknown_0
	unknown_1:5
	unknown_2:5
	unknown_3:5
	unknown_4:5
	unknown_5:5
	unknown_6:5
	unknown_7:55
	unknown_8:5
	unknown_9:5

unknown_10:5

unknown_11:5

unknown_12:5

unknown_13:5

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:

unknown_55:

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:

unknown_61:

unknown_62:
identity¢StatefulPartitionedCall¤	
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
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*L
_read_only_resource_inputs.
,*	
 !"%&'(+,-.1234789:=>?@*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_97_layer_call_and_return_conditional_losses_1001217o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¸
_input_shapes¦
£:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
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
1__inference_leaky_re_lu_911_layer_call_fn_1003766

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_911_layer_call_and_return_conditional_losses_1000532`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_dense_1007_layer_call_fn_1003562

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1007_layer_call_and_return_conditional_losses_1000480o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_913_layer_call_fn_1003925

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_913_layer_call_and_return_conditional_losses_1000253o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1007_layer_call_and_return_conditional_losses_1000480

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_911_layer_call_and_return_conditional_losses_1003761

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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
­
Ô
9__inference_batch_normalization_905_layer_call_fn_1003040

inputs
unknown:5
	unknown_0:5
	unknown_1:5
	unknown_2:5
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_905_layer_call_and_return_conditional_losses_999550o
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
Ê	
ø
G__inference_dense_1011_layer_call_and_return_conditional_losses_1004008

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_910_layer_call_and_return_conditional_losses_1003662

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1003_layer_call_and_return_conditional_losses_1000352

inputs0
matmul_readvariableop_resource:5-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
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
Ð
²
S__inference_batch_normalization_904_layer_call_and_return_conditional_losses_999468

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


/__inference_sequential_97_layer_call_fn_1000746
normalization_97_input
unknown
	unknown_0
	unknown_1:5
	unknown_2:5
	unknown_3:5
	unknown_4:5
	unknown_5:5
	unknown_6:5
	unknown_7:55
	unknown_8:5
	unknown_9:5

unknown_10:5

unknown_11:5

unknown_12:5

unknown_13:5

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:

unknown_46:

unknown_47:

unknown_48:

unknown_49:

unknown_50:

unknown_51:

unknown_52:

unknown_53:

unknown_54:

unknown_55:

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:

unknown_61:

unknown_62:
identity¢StatefulPartitionedCallÈ	
StatefulPartitionedCallStatefulPartitionedCallnormalization_97_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>?@*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_97_layer_call_and_return_conditional_losses_1000615o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¸
_input_shapes¦
£:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_97_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ê	
ø
G__inference_dense_1001_layer_call_and_return_conditional_losses_1002918

inputs0
matmul_readvariableop_resource:5-
biasadd_readvariableop_resource:5
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_912_layer_call_and_return_conditional_losses_1000124

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
Ô
9__inference_batch_normalization_904_layer_call_fn_1002931

inputs
unknown:5
	unknown_0:5
	unknown_1:5
	unknown_2:5
identity¢StatefulPartitionedCall
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_904_layer_call_and_return_conditional_losses_999468o
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
%
í
T__inference_batch_normalization_909_layer_call_and_return_conditional_losses_1003543

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_911_layer_call_and_return_conditional_losses_1000089

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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
Ê	
ø
G__inference_dense_1006_layer_call_and_return_conditional_losses_1003463

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
Ô
9__inference_batch_normalization_913_layer_call_fn_1003912

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_913_layer_call_and_return_conditional_losses_1000206o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_908_layer_call_and_return_conditional_losses_1003434

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1010_layer_call_and_return_conditional_losses_1000576

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_913_layer_call_and_return_conditional_losses_1003945

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1008_layer_call_and_return_conditional_losses_1003681

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_908_layer_call_fn_1003439

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_908_layer_call_and_return_conditional_losses_1000436`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
²
S__inference_batch_normalization_909_layer_call_and_return_conditional_losses_999878

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_912_layer_call_and_return_conditional_losses_1003870

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
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
­
M
1__inference_leaky_re_lu_906_layer_call_fn_1003221

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_906_layer_call_and_return_conditional_losses_1000372`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö©
®I
 __inference__traced_save_1004498
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	0
,savev2_dense_1001_kernel_read_readvariableop.
*savev2_dense_1001_bias_read_readvariableop<
8savev2_batch_normalization_904_gamma_read_readvariableop;
7savev2_batch_normalization_904_beta_read_readvariableopB
>savev2_batch_normalization_904_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_904_moving_variance_read_readvariableop0
,savev2_dense_1002_kernel_read_readvariableop.
*savev2_dense_1002_bias_read_readvariableop<
8savev2_batch_normalization_905_gamma_read_readvariableop;
7savev2_batch_normalization_905_beta_read_readvariableopB
>savev2_batch_normalization_905_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_905_moving_variance_read_readvariableop0
,savev2_dense_1003_kernel_read_readvariableop.
*savev2_dense_1003_bias_read_readvariableop<
8savev2_batch_normalization_906_gamma_read_readvariableop;
7savev2_batch_normalization_906_beta_read_readvariableopB
>savev2_batch_normalization_906_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_906_moving_variance_read_readvariableop0
,savev2_dense_1004_kernel_read_readvariableop.
*savev2_dense_1004_bias_read_readvariableop<
8savev2_batch_normalization_907_gamma_read_readvariableop;
7savev2_batch_normalization_907_beta_read_readvariableopB
>savev2_batch_normalization_907_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_907_moving_variance_read_readvariableop0
,savev2_dense_1005_kernel_read_readvariableop.
*savev2_dense_1005_bias_read_readvariableop<
8savev2_batch_normalization_908_gamma_read_readvariableop;
7savev2_batch_normalization_908_beta_read_readvariableopB
>savev2_batch_normalization_908_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_908_moving_variance_read_readvariableop0
,savev2_dense_1006_kernel_read_readvariableop.
*savev2_dense_1006_bias_read_readvariableop<
8savev2_batch_normalization_909_gamma_read_readvariableop;
7savev2_batch_normalization_909_beta_read_readvariableopB
>savev2_batch_normalization_909_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_909_moving_variance_read_readvariableop0
,savev2_dense_1007_kernel_read_readvariableop.
*savev2_dense_1007_bias_read_readvariableop<
8savev2_batch_normalization_910_gamma_read_readvariableop;
7savev2_batch_normalization_910_beta_read_readvariableopB
>savev2_batch_normalization_910_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_910_moving_variance_read_readvariableop0
,savev2_dense_1008_kernel_read_readvariableop.
*savev2_dense_1008_bias_read_readvariableop<
8savev2_batch_normalization_911_gamma_read_readvariableop;
7savev2_batch_normalization_911_beta_read_readvariableopB
>savev2_batch_normalization_911_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_911_moving_variance_read_readvariableop0
,savev2_dense_1009_kernel_read_readvariableop.
*savev2_dense_1009_bias_read_readvariableop<
8savev2_batch_normalization_912_gamma_read_readvariableop;
7savev2_batch_normalization_912_beta_read_readvariableopB
>savev2_batch_normalization_912_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_912_moving_variance_read_readvariableop0
,savev2_dense_1010_kernel_read_readvariableop.
*savev2_dense_1010_bias_read_readvariableop<
8savev2_batch_normalization_913_gamma_read_readvariableop;
7savev2_batch_normalization_913_beta_read_readvariableopB
>savev2_batch_normalization_913_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_913_moving_variance_read_readvariableop0
,savev2_dense_1011_kernel_read_readvariableop.
*savev2_dense_1011_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_dense_1001_kernel_m_read_readvariableop5
1savev2_adam_dense_1001_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_904_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_904_beta_m_read_readvariableop7
3savev2_adam_dense_1002_kernel_m_read_readvariableop5
1savev2_adam_dense_1002_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_905_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_905_beta_m_read_readvariableop7
3savev2_adam_dense_1003_kernel_m_read_readvariableop5
1savev2_adam_dense_1003_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_906_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_906_beta_m_read_readvariableop7
3savev2_adam_dense_1004_kernel_m_read_readvariableop5
1savev2_adam_dense_1004_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_907_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_907_beta_m_read_readvariableop7
3savev2_adam_dense_1005_kernel_m_read_readvariableop5
1savev2_adam_dense_1005_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_908_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_908_beta_m_read_readvariableop7
3savev2_adam_dense_1006_kernel_m_read_readvariableop5
1savev2_adam_dense_1006_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_909_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_909_beta_m_read_readvariableop7
3savev2_adam_dense_1007_kernel_m_read_readvariableop5
1savev2_adam_dense_1007_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_910_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_910_beta_m_read_readvariableop7
3savev2_adam_dense_1008_kernel_m_read_readvariableop5
1savev2_adam_dense_1008_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_911_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_911_beta_m_read_readvariableop7
3savev2_adam_dense_1009_kernel_m_read_readvariableop5
1savev2_adam_dense_1009_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_912_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_912_beta_m_read_readvariableop7
3savev2_adam_dense_1010_kernel_m_read_readvariableop5
1savev2_adam_dense_1010_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_913_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_913_beta_m_read_readvariableop7
3savev2_adam_dense_1011_kernel_m_read_readvariableop5
1savev2_adam_dense_1011_bias_m_read_readvariableop7
3savev2_adam_dense_1001_kernel_v_read_readvariableop5
1savev2_adam_dense_1001_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_904_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_904_beta_v_read_readvariableop7
3savev2_adam_dense_1002_kernel_v_read_readvariableop5
1savev2_adam_dense_1002_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_905_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_905_beta_v_read_readvariableop7
3savev2_adam_dense_1003_kernel_v_read_readvariableop5
1savev2_adam_dense_1003_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_906_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_906_beta_v_read_readvariableop7
3savev2_adam_dense_1004_kernel_v_read_readvariableop5
1savev2_adam_dense_1004_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_907_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_907_beta_v_read_readvariableop7
3savev2_adam_dense_1005_kernel_v_read_readvariableop5
1savev2_adam_dense_1005_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_908_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_908_beta_v_read_readvariableop7
3savev2_adam_dense_1006_kernel_v_read_readvariableop5
1savev2_adam_dense_1006_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_909_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_909_beta_v_read_readvariableop7
3savev2_adam_dense_1007_kernel_v_read_readvariableop5
1savev2_adam_dense_1007_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_910_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_910_beta_v_read_readvariableop7
3savev2_adam_dense_1008_kernel_v_read_readvariableop5
1savev2_adam_dense_1008_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_911_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_911_beta_v_read_readvariableop7
3savev2_adam_dense_1009_kernel_v_read_readvariableop5
1savev2_adam_dense_1009_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_912_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_912_beta_v_read_readvariableop7
3savev2_adam_dense_1010_kernel_v_read_readvariableop5
1savev2_adam_dense_1010_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_913_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_913_beta_v_read_readvariableop7
3savev2_adam_dense_1011_kernel_v_read_readvariableop5
1savev2_adam_dense_1011_bias_v_read_readvariableop
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
: ´W
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*ÜV
valueÒVBÏVB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-20/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-20/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-20/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-20/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-21/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHª
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*Î
valueÄBÁB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¤F
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop,savev2_dense_1001_kernel_read_readvariableop*savev2_dense_1001_bias_read_readvariableop8savev2_batch_normalization_904_gamma_read_readvariableop7savev2_batch_normalization_904_beta_read_readvariableop>savev2_batch_normalization_904_moving_mean_read_readvariableopBsavev2_batch_normalization_904_moving_variance_read_readvariableop,savev2_dense_1002_kernel_read_readvariableop*savev2_dense_1002_bias_read_readvariableop8savev2_batch_normalization_905_gamma_read_readvariableop7savev2_batch_normalization_905_beta_read_readvariableop>savev2_batch_normalization_905_moving_mean_read_readvariableopBsavev2_batch_normalization_905_moving_variance_read_readvariableop,savev2_dense_1003_kernel_read_readvariableop*savev2_dense_1003_bias_read_readvariableop8savev2_batch_normalization_906_gamma_read_readvariableop7savev2_batch_normalization_906_beta_read_readvariableop>savev2_batch_normalization_906_moving_mean_read_readvariableopBsavev2_batch_normalization_906_moving_variance_read_readvariableop,savev2_dense_1004_kernel_read_readvariableop*savev2_dense_1004_bias_read_readvariableop8savev2_batch_normalization_907_gamma_read_readvariableop7savev2_batch_normalization_907_beta_read_readvariableop>savev2_batch_normalization_907_moving_mean_read_readvariableopBsavev2_batch_normalization_907_moving_variance_read_readvariableop,savev2_dense_1005_kernel_read_readvariableop*savev2_dense_1005_bias_read_readvariableop8savev2_batch_normalization_908_gamma_read_readvariableop7savev2_batch_normalization_908_beta_read_readvariableop>savev2_batch_normalization_908_moving_mean_read_readvariableopBsavev2_batch_normalization_908_moving_variance_read_readvariableop,savev2_dense_1006_kernel_read_readvariableop*savev2_dense_1006_bias_read_readvariableop8savev2_batch_normalization_909_gamma_read_readvariableop7savev2_batch_normalization_909_beta_read_readvariableop>savev2_batch_normalization_909_moving_mean_read_readvariableopBsavev2_batch_normalization_909_moving_variance_read_readvariableop,savev2_dense_1007_kernel_read_readvariableop*savev2_dense_1007_bias_read_readvariableop8savev2_batch_normalization_910_gamma_read_readvariableop7savev2_batch_normalization_910_beta_read_readvariableop>savev2_batch_normalization_910_moving_mean_read_readvariableopBsavev2_batch_normalization_910_moving_variance_read_readvariableop,savev2_dense_1008_kernel_read_readvariableop*savev2_dense_1008_bias_read_readvariableop8savev2_batch_normalization_911_gamma_read_readvariableop7savev2_batch_normalization_911_beta_read_readvariableop>savev2_batch_normalization_911_moving_mean_read_readvariableopBsavev2_batch_normalization_911_moving_variance_read_readvariableop,savev2_dense_1009_kernel_read_readvariableop*savev2_dense_1009_bias_read_readvariableop8savev2_batch_normalization_912_gamma_read_readvariableop7savev2_batch_normalization_912_beta_read_readvariableop>savev2_batch_normalization_912_moving_mean_read_readvariableopBsavev2_batch_normalization_912_moving_variance_read_readvariableop,savev2_dense_1010_kernel_read_readvariableop*savev2_dense_1010_bias_read_readvariableop8savev2_batch_normalization_913_gamma_read_readvariableop7savev2_batch_normalization_913_beta_read_readvariableop>savev2_batch_normalization_913_moving_mean_read_readvariableopBsavev2_batch_normalization_913_moving_variance_read_readvariableop,savev2_dense_1011_kernel_read_readvariableop*savev2_dense_1011_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_dense_1001_kernel_m_read_readvariableop1savev2_adam_dense_1001_bias_m_read_readvariableop?savev2_adam_batch_normalization_904_gamma_m_read_readvariableop>savev2_adam_batch_normalization_904_beta_m_read_readvariableop3savev2_adam_dense_1002_kernel_m_read_readvariableop1savev2_adam_dense_1002_bias_m_read_readvariableop?savev2_adam_batch_normalization_905_gamma_m_read_readvariableop>savev2_adam_batch_normalization_905_beta_m_read_readvariableop3savev2_adam_dense_1003_kernel_m_read_readvariableop1savev2_adam_dense_1003_bias_m_read_readvariableop?savev2_adam_batch_normalization_906_gamma_m_read_readvariableop>savev2_adam_batch_normalization_906_beta_m_read_readvariableop3savev2_adam_dense_1004_kernel_m_read_readvariableop1savev2_adam_dense_1004_bias_m_read_readvariableop?savev2_adam_batch_normalization_907_gamma_m_read_readvariableop>savev2_adam_batch_normalization_907_beta_m_read_readvariableop3savev2_adam_dense_1005_kernel_m_read_readvariableop1savev2_adam_dense_1005_bias_m_read_readvariableop?savev2_adam_batch_normalization_908_gamma_m_read_readvariableop>savev2_adam_batch_normalization_908_beta_m_read_readvariableop3savev2_adam_dense_1006_kernel_m_read_readvariableop1savev2_adam_dense_1006_bias_m_read_readvariableop?savev2_adam_batch_normalization_909_gamma_m_read_readvariableop>savev2_adam_batch_normalization_909_beta_m_read_readvariableop3savev2_adam_dense_1007_kernel_m_read_readvariableop1savev2_adam_dense_1007_bias_m_read_readvariableop?savev2_adam_batch_normalization_910_gamma_m_read_readvariableop>savev2_adam_batch_normalization_910_beta_m_read_readvariableop3savev2_adam_dense_1008_kernel_m_read_readvariableop1savev2_adam_dense_1008_bias_m_read_readvariableop?savev2_adam_batch_normalization_911_gamma_m_read_readvariableop>savev2_adam_batch_normalization_911_beta_m_read_readvariableop3savev2_adam_dense_1009_kernel_m_read_readvariableop1savev2_adam_dense_1009_bias_m_read_readvariableop?savev2_adam_batch_normalization_912_gamma_m_read_readvariableop>savev2_adam_batch_normalization_912_beta_m_read_readvariableop3savev2_adam_dense_1010_kernel_m_read_readvariableop1savev2_adam_dense_1010_bias_m_read_readvariableop?savev2_adam_batch_normalization_913_gamma_m_read_readvariableop>savev2_adam_batch_normalization_913_beta_m_read_readvariableop3savev2_adam_dense_1011_kernel_m_read_readvariableop1savev2_adam_dense_1011_bias_m_read_readvariableop3savev2_adam_dense_1001_kernel_v_read_readvariableop1savev2_adam_dense_1001_bias_v_read_readvariableop?savev2_adam_batch_normalization_904_gamma_v_read_readvariableop>savev2_adam_batch_normalization_904_beta_v_read_readvariableop3savev2_adam_dense_1002_kernel_v_read_readvariableop1savev2_adam_dense_1002_bias_v_read_readvariableop?savev2_adam_batch_normalization_905_gamma_v_read_readvariableop>savev2_adam_batch_normalization_905_beta_v_read_readvariableop3savev2_adam_dense_1003_kernel_v_read_readvariableop1savev2_adam_dense_1003_bias_v_read_readvariableop?savev2_adam_batch_normalization_906_gamma_v_read_readvariableop>savev2_adam_batch_normalization_906_beta_v_read_readvariableop3savev2_adam_dense_1004_kernel_v_read_readvariableop1savev2_adam_dense_1004_bias_v_read_readvariableop?savev2_adam_batch_normalization_907_gamma_v_read_readvariableop>savev2_adam_batch_normalization_907_beta_v_read_readvariableop3savev2_adam_dense_1005_kernel_v_read_readvariableop1savev2_adam_dense_1005_bias_v_read_readvariableop?savev2_adam_batch_normalization_908_gamma_v_read_readvariableop>savev2_adam_batch_normalization_908_beta_v_read_readvariableop3savev2_adam_dense_1006_kernel_v_read_readvariableop1savev2_adam_dense_1006_bias_v_read_readvariableop?savev2_adam_batch_normalization_909_gamma_v_read_readvariableop>savev2_adam_batch_normalization_909_beta_v_read_readvariableop3savev2_adam_dense_1007_kernel_v_read_readvariableop1savev2_adam_dense_1007_bias_v_read_readvariableop?savev2_adam_batch_normalization_910_gamma_v_read_readvariableop>savev2_adam_batch_normalization_910_beta_v_read_readvariableop3savev2_adam_dense_1008_kernel_v_read_readvariableop1savev2_adam_dense_1008_bias_v_read_readvariableop?savev2_adam_batch_normalization_911_gamma_v_read_readvariableop>savev2_adam_batch_normalization_911_beta_v_read_readvariableop3savev2_adam_dense_1009_kernel_v_read_readvariableop1savev2_adam_dense_1009_bias_v_read_readvariableop?savev2_adam_batch_normalization_912_gamma_v_read_readvariableop>savev2_adam_batch_normalization_912_beta_v_read_readvariableop3savev2_adam_dense_1010_kernel_v_read_readvariableop1savev2_adam_dense_1010_bias_v_read_readvariableop?savev2_adam_batch_normalization_913_gamma_v_read_readvariableop>savev2_adam_batch_normalization_913_beta_v_read_readvariableop3savev2_adam_dense_1011_kernel_v_read_readvariableop1savev2_adam_dense_1011_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *­
dtypes¢
2		
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

identity_1Identity_1:output:0*£
_input_shapes
: ::: :5:5:5:5:5:5:55:5:5:5:5:5:5:::::::::::::::::::::::::::::::::::::::::::::::::: : : : : : :5:5:5:5:55:5:5:5:5::::::::::::::::::::::::::::::::::5:5:5:5:55:5:5:5:5:::::::::::::::::::::::::::::::::: 2(
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

:5: 

_output_shapes
:5: 

_output_shapes
:5: 

_output_shapes
:5: 

_output_shapes
:5: 	

_output_shapes
:5:$
 

_output_shapes

:55: 

_output_shapes
:5: 

_output_shapes
:5: 

_output_shapes
:5: 

_output_shapes
:5: 

_output_shapes
:5:$ 

_output_shapes

:5: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
::$. 

_output_shapes

:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
::$4 

_output_shapes

:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
::$: 

_output_shapes

:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
:: >

_output_shapes
:: ?

_output_shapes
::$@ 

_output_shapes

:: A

_output_shapes
::B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :F

_output_shapes
: :G

_output_shapes
: :$H 

_output_shapes

:5: I
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

:55: M

_output_shapes
:5: N

_output_shapes
:5: O

_output_shapes
:5:$P 

_output_shapes

:5: Q

_output_shapes
:: R

_output_shapes
:: S

_output_shapes
::$T 

_output_shapes

:: U

_output_shapes
:: V

_output_shapes
:: W

_output_shapes
::$X 

_output_shapes

:: Y

_output_shapes
:: Z

_output_shapes
:: [

_output_shapes
::$\ 

_output_shapes

:: ]

_output_shapes
:: ^

_output_shapes
:: _

_output_shapes
::$` 

_output_shapes

:: a

_output_shapes
:: b

_output_shapes
:: c

_output_shapes
::$d 

_output_shapes

:: e

_output_shapes
:: f

_output_shapes
:: g

_output_shapes
::$h 

_output_shapes

:: i

_output_shapes
:: j

_output_shapes
:: k

_output_shapes
::$l 

_output_shapes

:: m

_output_shapes
:: n

_output_shapes
:: o

_output_shapes
::$p 

_output_shapes

:: q

_output_shapes
::$r 

_output_shapes

:5: s

_output_shapes
:5: t

_output_shapes
:5: u

_output_shapes
:5:$v 

_output_shapes

:55: w

_output_shapes
:5: x

_output_shapes
:5: y

_output_shapes
:5:$z 

_output_shapes

:5: {

_output_shapes
:: |

_output_shapes
:: }

_output_shapes
::$~ 

_output_shapes

:: 

_output_shapes
::!

_output_shapes
::!

_output_shapes
::% 

_output_shapes

::!

_output_shapes
::!

_output_shapes
::!

_output_shapes
::% 

_output_shapes

::!

_output_shapes
::!

_output_shapes
::!

_output_shapes
::% 

_output_shapes

::!

_output_shapes
::!

_output_shapes
::!

_output_shapes
::% 

_output_shapes

::!

_output_shapes
::!

_output_shapes
::!

_output_shapes
::% 

_output_shapes

::!

_output_shapes
::!

_output_shapes
::!

_output_shapes
::% 

_output_shapes

::!

_output_shapes
::!

_output_shapes
::!

_output_shapes
::% 

_output_shapes

::!

_output_shapes
::

_output_shapes
: 
æ
h
L__inference_leaky_re_lu_910_layer_call_and_return_conditional_losses_1000500

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾§

J__inference_sequential_97_layer_call_and_return_conditional_losses_1000615

inputs
normalization_97_sub_y
normalization_97_sqrt_x$
dense_1001_1000289:5 
dense_1001_1000291:5-
batch_normalization_904_1000294:5-
batch_normalization_904_1000296:5-
batch_normalization_904_1000298:5-
batch_normalization_904_1000300:5$
dense_1002_1000321:55 
dense_1002_1000323:5-
batch_normalization_905_1000326:5-
batch_normalization_905_1000328:5-
batch_normalization_905_1000330:5-
batch_normalization_905_1000332:5$
dense_1003_1000353:5 
dense_1003_1000355:-
batch_normalization_906_1000358:-
batch_normalization_906_1000360:-
batch_normalization_906_1000362:-
batch_normalization_906_1000364:$
dense_1004_1000385: 
dense_1004_1000387:-
batch_normalization_907_1000390:-
batch_normalization_907_1000392:-
batch_normalization_907_1000394:-
batch_normalization_907_1000396:$
dense_1005_1000417: 
dense_1005_1000419:-
batch_normalization_908_1000422:-
batch_normalization_908_1000424:-
batch_normalization_908_1000426:-
batch_normalization_908_1000428:$
dense_1006_1000449: 
dense_1006_1000451:-
batch_normalization_909_1000454:-
batch_normalization_909_1000456:-
batch_normalization_909_1000458:-
batch_normalization_909_1000460:$
dense_1007_1000481: 
dense_1007_1000483:-
batch_normalization_910_1000486:-
batch_normalization_910_1000488:-
batch_normalization_910_1000490:-
batch_normalization_910_1000492:$
dense_1008_1000513: 
dense_1008_1000515:-
batch_normalization_911_1000518:-
batch_normalization_911_1000520:-
batch_normalization_911_1000522:-
batch_normalization_911_1000524:$
dense_1009_1000545: 
dense_1009_1000547:-
batch_normalization_912_1000550:-
batch_normalization_912_1000552:-
batch_normalization_912_1000554:-
batch_normalization_912_1000556:$
dense_1010_1000577: 
dense_1010_1000579:-
batch_normalization_913_1000582:-
batch_normalization_913_1000584:-
batch_normalization_913_1000586:-
batch_normalization_913_1000588:$
dense_1011_1000609: 
dense_1011_1000611:
identity¢/batch_normalization_904/StatefulPartitionedCall¢/batch_normalization_905/StatefulPartitionedCall¢/batch_normalization_906/StatefulPartitionedCall¢/batch_normalization_907/StatefulPartitionedCall¢/batch_normalization_908/StatefulPartitionedCall¢/batch_normalization_909/StatefulPartitionedCall¢/batch_normalization_910/StatefulPartitionedCall¢/batch_normalization_911/StatefulPartitionedCall¢/batch_normalization_912/StatefulPartitionedCall¢/batch_normalization_913/StatefulPartitionedCall¢"dense_1001/StatefulPartitionedCall¢"dense_1002/StatefulPartitionedCall¢"dense_1003/StatefulPartitionedCall¢"dense_1004/StatefulPartitionedCall¢"dense_1005/StatefulPartitionedCall¢"dense_1006/StatefulPartitionedCall¢"dense_1007/StatefulPartitionedCall¢"dense_1008/StatefulPartitionedCall¢"dense_1009/StatefulPartitionedCall¢"dense_1010/StatefulPartitionedCall¢"dense_1011/StatefulPartitionedCallm
normalization_97/subSubinputsnormalization_97_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_97/SqrtSqrtnormalization_97_sqrt_x*
T0*
_output_shapes

:_
normalization_97/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_97/MaximumMaximumnormalization_97/Sqrt:y:0#normalization_97/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_97/truedivRealDivnormalization_97/sub:z:0normalization_97/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"dense_1001/StatefulPartitionedCallStatefulPartitionedCallnormalization_97/truediv:z:0dense_1001_1000289dense_1001_1000291*
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
GPU 2J 8 *P
fKRI
G__inference_dense_1001_layer_call_and_return_conditional_losses_1000288
/batch_normalization_904/StatefulPartitionedCallStatefulPartitionedCall+dense_1001/StatefulPartitionedCall:output:0batch_normalization_904_1000294batch_normalization_904_1000296batch_normalization_904_1000298batch_normalization_904_1000300*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_904_layer_call_and_return_conditional_losses_999468ù
leaky_re_lu_904/PartitionedCallPartitionedCall8batch_normalization_904/StatefulPartitionedCall:output:0*
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
L__inference_leaky_re_lu_904_layer_call_and_return_conditional_losses_1000308
"dense_1002/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_904/PartitionedCall:output:0dense_1002_1000321dense_1002_1000323*
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
GPU 2J 8 *P
fKRI
G__inference_dense_1002_layer_call_and_return_conditional_losses_1000320
/batch_normalization_905/StatefulPartitionedCallStatefulPartitionedCall+dense_1002/StatefulPartitionedCall:output:0batch_normalization_905_1000326batch_normalization_905_1000328batch_normalization_905_1000330batch_normalization_905_1000332*
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
GPU 2J 8 *\
fWRU
S__inference_batch_normalization_905_layer_call_and_return_conditional_losses_999550ù
leaky_re_lu_905/PartitionedCallPartitionedCall8batch_normalization_905/StatefulPartitionedCall:output:0*
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
L__inference_leaky_re_lu_905_layer_call_and_return_conditional_losses_1000340
"dense_1003/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_905/PartitionedCall:output:0dense_1003_1000353dense_1003_1000355*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1003_layer_call_and_return_conditional_losses_1000352
/batch_normalization_906/StatefulPartitionedCallStatefulPartitionedCall+dense_1003/StatefulPartitionedCall:output:0batch_normalization_906_1000358batch_normalization_906_1000360batch_normalization_906_1000362batch_normalization_906_1000364*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_906_layer_call_and_return_conditional_losses_999632ù
leaky_re_lu_906/PartitionedCallPartitionedCall8batch_normalization_906/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_906_layer_call_and_return_conditional_losses_1000372
"dense_1004/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_906/PartitionedCall:output:0dense_1004_1000385dense_1004_1000387*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1004_layer_call_and_return_conditional_losses_1000384
/batch_normalization_907/StatefulPartitionedCallStatefulPartitionedCall+dense_1004/StatefulPartitionedCall:output:0batch_normalization_907_1000390batch_normalization_907_1000392batch_normalization_907_1000394batch_normalization_907_1000396*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_907_layer_call_and_return_conditional_losses_999714ù
leaky_re_lu_907/PartitionedCallPartitionedCall8batch_normalization_907/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_907_layer_call_and_return_conditional_losses_1000404
"dense_1005/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_907/PartitionedCall:output:0dense_1005_1000417dense_1005_1000419*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1005_layer_call_and_return_conditional_losses_1000416
/batch_normalization_908/StatefulPartitionedCallStatefulPartitionedCall+dense_1005/StatefulPartitionedCall:output:0batch_normalization_908_1000422batch_normalization_908_1000424batch_normalization_908_1000426batch_normalization_908_1000428*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_908_layer_call_and_return_conditional_losses_999796ù
leaky_re_lu_908/PartitionedCallPartitionedCall8batch_normalization_908/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_908_layer_call_and_return_conditional_losses_1000436
"dense_1006/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_908/PartitionedCall:output:0dense_1006_1000449dense_1006_1000451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1006_layer_call_and_return_conditional_losses_1000448
/batch_normalization_909/StatefulPartitionedCallStatefulPartitionedCall+dense_1006/StatefulPartitionedCall:output:0batch_normalization_909_1000454batch_normalization_909_1000456batch_normalization_909_1000458batch_normalization_909_1000460*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_909_layer_call_and_return_conditional_losses_999878ù
leaky_re_lu_909/PartitionedCallPartitionedCall8batch_normalization_909/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_909_layer_call_and_return_conditional_losses_1000468
"dense_1007/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_909/PartitionedCall:output:0dense_1007_1000481dense_1007_1000483*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1007_layer_call_and_return_conditional_losses_1000480
/batch_normalization_910/StatefulPartitionedCallStatefulPartitionedCall+dense_1007/StatefulPartitionedCall:output:0batch_normalization_910_1000486batch_normalization_910_1000488batch_normalization_910_1000490batch_normalization_910_1000492*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_910_layer_call_and_return_conditional_losses_999960ù
leaky_re_lu_910/PartitionedCallPartitionedCall8batch_normalization_910/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_910_layer_call_and_return_conditional_losses_1000500
"dense_1008/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_910/PartitionedCall:output:0dense_1008_1000513dense_1008_1000515*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1008_layer_call_and_return_conditional_losses_1000512
/batch_normalization_911/StatefulPartitionedCallStatefulPartitionedCall+dense_1008/StatefulPartitionedCall:output:0batch_normalization_911_1000518batch_normalization_911_1000520batch_normalization_911_1000522batch_normalization_911_1000524*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_911_layer_call_and_return_conditional_losses_1000042ù
leaky_re_lu_911/PartitionedCallPartitionedCall8batch_normalization_911/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_911_layer_call_and_return_conditional_losses_1000532
"dense_1009/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_911/PartitionedCall:output:0dense_1009_1000545dense_1009_1000547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1009_layer_call_and_return_conditional_losses_1000544
/batch_normalization_912/StatefulPartitionedCallStatefulPartitionedCall+dense_1009/StatefulPartitionedCall:output:0batch_normalization_912_1000550batch_normalization_912_1000552batch_normalization_912_1000554batch_normalization_912_1000556*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_912_layer_call_and_return_conditional_losses_1000124ù
leaky_re_lu_912/PartitionedCallPartitionedCall8batch_normalization_912/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_912_layer_call_and_return_conditional_losses_1000564
"dense_1010/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_912/PartitionedCall:output:0dense_1010_1000577dense_1010_1000579*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1010_layer_call_and_return_conditional_losses_1000576
/batch_normalization_913/StatefulPartitionedCallStatefulPartitionedCall+dense_1010/StatefulPartitionedCall:output:0batch_normalization_913_1000582batch_normalization_913_1000584batch_normalization_913_1000586batch_normalization_913_1000588*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_913_layer_call_and_return_conditional_losses_1000206ù
leaky_re_lu_913/PartitionedCallPartitionedCall8batch_normalization_913/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_913_layer_call_and_return_conditional_losses_1000596
"dense_1011/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_913/PartitionedCall:output:0dense_1011_1000609dense_1011_1000611*
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
GPU 2J 8 *P
fKRI
G__inference_dense_1011_layer_call_and_return_conditional_losses_1000608z
IdentityIdentity+dense_1011/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp0^batch_normalization_904/StatefulPartitionedCall0^batch_normalization_905/StatefulPartitionedCall0^batch_normalization_906/StatefulPartitionedCall0^batch_normalization_907/StatefulPartitionedCall0^batch_normalization_908/StatefulPartitionedCall0^batch_normalization_909/StatefulPartitionedCall0^batch_normalization_910/StatefulPartitionedCall0^batch_normalization_911/StatefulPartitionedCall0^batch_normalization_912/StatefulPartitionedCall0^batch_normalization_913/StatefulPartitionedCall#^dense_1001/StatefulPartitionedCall#^dense_1002/StatefulPartitionedCall#^dense_1003/StatefulPartitionedCall#^dense_1004/StatefulPartitionedCall#^dense_1005/StatefulPartitionedCall#^dense_1006/StatefulPartitionedCall#^dense_1007/StatefulPartitionedCall#^dense_1008/StatefulPartitionedCall#^dense_1009/StatefulPartitionedCall#^dense_1010/StatefulPartitionedCall#^dense_1011/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¸
_input_shapes¦
£:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_904/StatefulPartitionedCall/batch_normalization_904/StatefulPartitionedCall2b
/batch_normalization_905/StatefulPartitionedCall/batch_normalization_905/StatefulPartitionedCall2b
/batch_normalization_906/StatefulPartitionedCall/batch_normalization_906/StatefulPartitionedCall2b
/batch_normalization_907/StatefulPartitionedCall/batch_normalization_907/StatefulPartitionedCall2b
/batch_normalization_908/StatefulPartitionedCall/batch_normalization_908/StatefulPartitionedCall2b
/batch_normalization_909/StatefulPartitionedCall/batch_normalization_909/StatefulPartitionedCall2b
/batch_normalization_910/StatefulPartitionedCall/batch_normalization_910/StatefulPartitionedCall2b
/batch_normalization_911/StatefulPartitionedCall/batch_normalization_911/StatefulPartitionedCall2b
/batch_normalization_912/StatefulPartitionedCall/batch_normalization_912/StatefulPartitionedCall2b
/batch_normalization_913/StatefulPartitionedCall/batch_normalization_913/StatefulPartitionedCall2H
"dense_1001/StatefulPartitionedCall"dense_1001/StatefulPartitionedCall2H
"dense_1002/StatefulPartitionedCall"dense_1002/StatefulPartitionedCall2H
"dense_1003/StatefulPartitionedCall"dense_1003/StatefulPartitionedCall2H
"dense_1004/StatefulPartitionedCall"dense_1004/StatefulPartitionedCall2H
"dense_1005/StatefulPartitionedCall"dense_1005/StatefulPartitionedCall2H
"dense_1006/StatefulPartitionedCall"dense_1006/StatefulPartitionedCall2H
"dense_1007/StatefulPartitionedCall"dense_1007/StatefulPartitionedCall2H
"dense_1008/StatefulPartitionedCall"dense_1008/StatefulPartitionedCall2H
"dense_1009/StatefulPartitionedCall"dense_1009/StatefulPartitionedCall2H
"dense_1010/StatefulPartitionedCall"dense_1010/StatefulPartitionedCall2H
"dense_1011/StatefulPartitionedCall"dense_1011/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ñ
³
T__inference_batch_normalization_909_layer_call_and_return_conditional_losses_1003509

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_907_layer_call_and_return_conditional_losses_1000404

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_911_layer_call_and_return_conditional_losses_1000042

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_913_layer_call_and_return_conditional_losses_1000596

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_908_layer_call_and_return_conditional_losses_1000436

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_905_layer_call_and_return_conditional_losses_1000340

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
Ê	
ø
G__inference_dense_1009_layer_call_and_return_conditional_losses_1000544

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1005_layer_call_and_return_conditional_losses_1000416

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1002_layer_call_and_return_conditional_losses_1000320

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
T__inference_batch_normalization_905_layer_call_and_return_conditional_losses_1003107

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
%
ì
S__inference_batch_normalization_906_layer_call_and_return_conditional_losses_999679

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_912_layer_call_and_return_conditional_losses_1003836

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_910_layer_call_and_return_conditional_losses_1003618

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï'
Ó
__inference_adapt_step_1002899
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
È

,__inference_dense_1002_layer_call_fn_1003017

inputs
unknown:55
	unknown_0:5
identity¢StatefulPartitionedCallÜ
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
GPU 2J 8 *P
fKRI
G__inference_dense_1002_layer_call_and_return_conditional_losses_1000320o
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
­
Ô
9__inference_batch_normalization_907_layer_call_fn_1003258

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_907_layer_call_and_return_conditional_losses_999714o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_dense_1005_layer_call_fn_1003344

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_1005_layer_call_and_return_conditional_losses_1000416o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_dense_1004_layer_call_and_return_conditional_losses_1003245

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_907_layer_call_fn_1003330

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_907_layer_call_and_return_conditional_losses_1000404`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ë
serving_default·
Y
normalization_97_input?
(serving_default_normalization_97_input:0ÿÿÿÿÿÿÿÿÿ>

dense_10110
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:­
ä	
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
!	optimizer
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(_default_save_signature
)
signatures"
_tf_keras_sequential
Ó
*
_keep_axis
+_reduce_axis
,_reduce_axis_mask
-_broadcast_shape
.mean
.
adapt_mean
/variance
/adapt_variance
	0count
1	keras_api
2_adapt_function"
_tf_keras_layer
»

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Taxis
	Ugamma
Vbeta
Wmoving_mean
Xmoving_variance
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
»

ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
maxis
	ngamma
obeta
pmoving_mean
qmoving_variance
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

~kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

 gamma
	¡beta
¢moving_mean
£moving_variance
¤	variables
¥trainable_variables
¦regularization_losses
§	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
°kernel
	±bias
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	¸axis

¹gamma
	ºbeta
»moving_mean
¼moving_variance
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ékernel
	Êbias
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	Ñaxis

Ògamma
	Óbeta
Ômoving_mean
Õmoving_variance
Ö	variables
×trainable_variables
Øregularization_losses
Ù	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ü	variables
Ýtrainable_variables
Þregularization_losses
ß	keras_api
à__call__
+á&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
âkernel
	ãbias
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
è__call__
+é&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	êaxis

ëgamma
	ìbeta
ímoving_mean
îmoving_variance
ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"
_tf_keras_layer
«
õ	variables
ötrainable_variables
÷regularization_losses
ø	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
ûkernel
	übias
ý	variables
þtrainable_variables
ÿregularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
 moving_variance
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
«
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
­kernel
	®bias
¯	variables
°trainable_variables
±regularization_losses
²	keras_api
³__call__
+´&call_and_return_all_conditional_losses"
_tf_keras_layer
Ä
	µiter
¶beta_1
·beta_2

¸decay3mß4mà<má=mâLmãMmäUmåVmæemçfmènméomê~mëmì	mí	mî	mï	mð	 mñ	¡mò	°mó	±mô	¹mõ	ºmö	Ém÷	Êmø	Òmù	Ómú	âmû	ãmü	ëmý	ìmþ	ûmÿ	üm	m	m	m	m	m	m	­m	®m3v4v<v=vLvMvUvVvevfvnvov~vv	v	v	v	v	 v	¡v	°v	±v	¹v	ºv 	Év¡	Êv¢	Òv£	Óv¤	âv¥	ãv¦	ëv§	ìv¨	ûv©	üvª	v«	v¬	v­	v®	v¯	v°	­v±	®v²"
	optimizer
È
.0
/1
02
33
44
<5
=6
>7
?8
L9
M10
U11
V12
W13
X14
e15
f16
n17
o18
p19
q20
~21
22
23
24
25
26
27
28
 29
¡30
¢31
£32
°33
±34
¹35
º36
»37
¼38
É39
Ê40
Ò41
Ó42
Ô43
Õ44
â45
ã46
ë47
ì48
í49
î50
û51
ü52
53
54
55
56
57
58
59
60
61
 62
­63
®64"
trackable_list_wrapper

30
41
<2
=3
L4
M5
U6
V7
e8
f9
n10
o11
~12
13
14
15
16
17
 18
¡19
°20
±21
¹22
º23
É24
Ê25
Ò26
Ó27
â28
ã29
ë30
ì31
û32
ü33
34
35
36
37
38
39
­40
®41"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
(_default_save_signature
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
2
/__inference_sequential_97_layer_call_fn_1000746
/__inference_sequential_97_layer_call_fn_1001950
/__inference_sequential_97_layer_call_fn_1002083
/__inference_sequential_97_layer_call_fn_1001481À
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
J__inference_sequential_97_layer_call_and_return_conditional_losses_1002330
J__inference_sequential_97_layer_call_and_return_conditional_losses_1002717
J__inference_sequential_97_layer_call_and_return_conditional_losses_1001647
J__inference_sequential_97_layer_call_and_return_conditional_losses_1001813À
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
!__inference__wrapped_model_999444normalization_97_input"
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
¾serving_default"
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
__inference_adapt_step_1002899
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
#:!52dense_1001/kernel
:52dense_1001/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_dense_1001_layer_call_fn_1002908¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_1001_layer_call_and_return_conditional_losses_1002918¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)52batch_normalization_904/gamma
*:(52batch_normalization_904/beta
3:15 (2#batch_normalization_904/moving_mean
7:55 (2'batch_normalization_904/moving_variance
<
<0
=1
>2
?3"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Änon_trainable_variables
Ålayers
Æmetrics
 Çlayer_regularization_losses
Èlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_904_layer_call_fn_1002931
9__inference_batch_normalization_904_layer_call_fn_1002944´
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
T__inference_batch_normalization_904_layer_call_and_return_conditional_losses_1002964
T__inference_batch_normalization_904_layer_call_and_return_conditional_losses_1002998´
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
Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_904_layer_call_fn_1003003¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
L__inference_leaky_re_lu_904_layer_call_and_return_conditional_losses_1003008¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!552dense_1002/kernel
:52dense_1002/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Înon_trainable_variables
Ïlayers
Ðmetrics
 Ñlayer_regularization_losses
Òlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_dense_1002_layer_call_fn_1003017¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_1002_layer_call_and_return_conditional_losses_1003027¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)52batch_normalization_905/gamma
*:(52batch_normalization_905/beta
3:15 (2#batch_normalization_905/moving_mean
7:55 (2'batch_normalization_905/moving_variance
<
U0
V1
W2
X3"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ónon_trainable_variables
Ôlayers
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_905_layer_call_fn_1003040
9__inference_batch_normalization_905_layer_call_fn_1003053´
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
T__inference_batch_normalization_905_layer_call_and_return_conditional_losses_1003073
T__inference_batch_normalization_905_layer_call_and_return_conditional_losses_1003107´
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
Ønon_trainable_variables
Ùlayers
Úmetrics
 Ûlayer_regularization_losses
Ülayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_905_layer_call_fn_1003112¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
L__inference_leaky_re_lu_905_layer_call_and_return_conditional_losses_1003117¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!52dense_1003/kernel
:2dense_1003/bias
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_dense_1003_layer_call_fn_1003126¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_1003_layer_call_and_return_conditional_losses_1003136¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)2batch_normalization_906/gamma
*:(2batch_normalization_906/beta
3:1 (2#batch_normalization_906/moving_mean
7:5 (2'batch_normalization_906/moving_variance
<
n0
o1
p2
q3"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_906_layer_call_fn_1003149
9__inference_batch_normalization_906_layer_call_fn_1003162´
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
T__inference_batch_normalization_906_layer_call_and_return_conditional_losses_1003182
T__inference_batch_normalization_906_layer_call_and_return_conditional_losses_1003216´
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
çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_906_layer_call_fn_1003221¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
L__inference_leaky_re_lu_906_layer_call_and_return_conditional_losses_1003226¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!2dense_1004/kernel
:2dense_1004/bias
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_dense_1004_layer_call_fn_1003235¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_1004_layer_call_and_return_conditional_losses_1003245¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)2batch_normalization_907/gamma
*:(2batch_normalization_907/beta
3:1 (2#batch_normalization_907/moving_mean
7:5 (2'batch_normalization_907/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ñnon_trainable_variables
òlayers
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_907_layer_call_fn_1003258
9__inference_batch_normalization_907_layer_call_fn_1003271´
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
T__inference_batch_normalization_907_layer_call_and_return_conditional_losses_1003291
T__inference_batch_normalization_907_layer_call_and_return_conditional_losses_1003325´
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
önon_trainable_variables
÷layers
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_907_layer_call_fn_1003330¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
L__inference_leaky_re_lu_907_layer_call_and_return_conditional_losses_1003335¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!2dense_1005/kernel
:2dense_1005/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_dense_1005_layer_call_fn_1003344¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_1005_layer_call_and_return_conditional_losses_1003354¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)2batch_normalization_908/gamma
*:(2batch_normalization_908/beta
3:1 (2#batch_normalization_908/moving_mean
7:5 (2'batch_normalization_908/moving_variance
@
 0
¡1
¢2
£3"
trackable_list_wrapper
0
 0
¡1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¤	variables
¥trainable_variables
¦regularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_908_layer_call_fn_1003367
9__inference_batch_normalization_908_layer_call_fn_1003380´
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
T__inference_batch_normalization_908_layer_call_and_return_conditional_losses_1003400
T__inference_batch_normalization_908_layer_call_and_return_conditional_losses_1003434´
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_908_layer_call_fn_1003439¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
L__inference_leaky_re_lu_908_layer_call_and_return_conditional_losses_1003444¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!2dense_1006/kernel
:2dense_1006/bias
0
°0
±1"
trackable_list_wrapper
0
°0
±1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_dense_1006_layer_call_fn_1003453¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_1006_layer_call_and_return_conditional_losses_1003463¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)2batch_normalization_909/gamma
*:(2batch_normalization_909/beta
3:1 (2#batch_normalization_909/moving_mean
7:5 (2'batch_normalization_909/moving_variance
@
¹0
º1
»2
¼3"
trackable_list_wrapper
0
¹0
º1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_909_layer_call_fn_1003476
9__inference_batch_normalization_909_layer_call_fn_1003489´
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
T__inference_batch_normalization_909_layer_call_and_return_conditional_losses_1003509
T__inference_batch_normalization_909_layer_call_and_return_conditional_losses_1003543´
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_909_layer_call_fn_1003548¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
L__inference_leaky_re_lu_909_layer_call_and_return_conditional_losses_1003553¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!2dense_1007/kernel
:2dense_1007/bias
0
É0
Ê1"
trackable_list_wrapper
0
É0
Ê1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_dense_1007_layer_call_fn_1003562¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_1007_layer_call_and_return_conditional_losses_1003572¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)2batch_normalization_910/gamma
*:(2batch_normalization_910/beta
3:1 (2#batch_normalization_910/moving_mean
7:5 (2'batch_normalization_910/moving_variance
@
Ò0
Ó1
Ô2
Õ3"
trackable_list_wrapper
0
Ò0
Ó1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
Ö	variables
×trainable_variables
Øregularization_losses
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_910_layer_call_fn_1003585
9__inference_batch_normalization_910_layer_call_fn_1003598´
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
T__inference_batch_normalization_910_layer_call_and_return_conditional_losses_1003618
T__inference_batch_normalization_910_layer_call_and_return_conditional_losses_1003652´
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
£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
Ü	variables
Ýtrainable_variables
Þregularization_losses
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_910_layer_call_fn_1003657¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
L__inference_leaky_re_lu_910_layer_call_and_return_conditional_losses_1003662¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!2dense_1008/kernel
:2dense_1008/bias
0
â0
ã1"
trackable_list_wrapper
0
â0
ã1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
ä	variables
åtrainable_variables
æregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_dense_1008_layer_call_fn_1003671¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_1008_layer_call_and_return_conditional_losses_1003681¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)2batch_normalization_911/gamma
*:(2batch_normalization_911/beta
3:1 (2#batch_normalization_911/moving_mean
7:5 (2'batch_normalization_911/moving_variance
@
ë0
ì1
í2
î3"
trackable_list_wrapper
0
ë0
ì1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
ï	variables
ðtrainable_variables
ñregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_911_layer_call_fn_1003694
9__inference_batch_normalization_911_layer_call_fn_1003707´
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
T__inference_batch_normalization_911_layer_call_and_return_conditional_losses_1003727
T__inference_batch_normalization_911_layer_call_and_return_conditional_losses_1003761´
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
²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
õ	variables
ötrainable_variables
÷regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_911_layer_call_fn_1003766¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
L__inference_leaky_re_lu_911_layer_call_and_return_conditional_losses_1003771¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!2dense_1009/kernel
:2dense_1009/bias
0
û0
ü1"
trackable_list_wrapper
0
û0
ü1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
ý	variables
þtrainable_variables
ÿregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_dense_1009_layer_call_fn_1003780¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_1009_layer_call_and_return_conditional_losses_1003790¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)2batch_normalization_912/gamma
*:(2batch_normalization_912/beta
3:1 (2#batch_normalization_912/moving_mean
7:5 (2'batch_normalization_912/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_912_layer_call_fn_1003803
9__inference_batch_normalization_912_layer_call_fn_1003816´
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
T__inference_batch_normalization_912_layer_call_and_return_conditional_losses_1003836
T__inference_batch_normalization_912_layer_call_and_return_conditional_losses_1003870´
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
Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_912_layer_call_fn_1003875¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
L__inference_leaky_re_lu_912_layer_call_and_return_conditional_losses_1003880¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!2dense_1010/kernel
:2dense_1010/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_dense_1010_layer_call_fn_1003889¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_1010_layer_call_and_return_conditional_losses_1003899¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)2batch_normalization_913/gamma
*:(2batch_normalization_913/beta
3:1 (2#batch_normalization_913/moving_mean
7:5 (2'batch_normalization_913/moving_variance
@
0
1
2
 3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_913_layer_call_fn_1003912
9__inference_batch_normalization_913_layer_call_fn_1003925´
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
T__inference_batch_normalization_913_layer_call_and_return_conditional_losses_1003945
T__inference_batch_normalization_913_layer_call_and_return_conditional_losses_1003979´
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
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_913_layer_call_fn_1003984¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
L__inference_leaky_re_lu_913_layer_call_and_return_conditional_losses_1003989¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!2dense_1011/kernel
:2dense_1011/bias
0
­0
®1"
trackable_list_wrapper
0
­0
®1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
¯	variables
°trainable_variables
±regularization_losses
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_dense_1011_layer_call_fn_1003998¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_1011_layer_call_and_return_conditional_losses_1004008¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
Ü
.0
/1
02
>3
?4
W5
X6
p7
q8
9
10
¢11
£12
»13
¼14
Ô15
Õ16
í17
î18
19
20
21
 22"
trackable_list_wrapper

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
 31"
trackable_list_wrapper
(
Ú0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
%__inference_signature_wrapper_1002852normalization_97_input"
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
>0
?1"
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
W0
X1"
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
p0
q1"
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
0
1"
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
¢0
£1"
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
»0
¼1"
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
Ô0
Õ1"
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
í0
î1"
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
0
1"
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
0
 1"
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

Ûtotal

Ücount
Ý	variables
Þ	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
Û0
Ü1"
trackable_list_wrapper
.
Ý	variables"
_generic_user_object
(:&52Adam/dense_1001/kernel/m
": 52Adam/dense_1001/bias/m
0:.52$Adam/batch_normalization_904/gamma/m
/:-52#Adam/batch_normalization_904/beta/m
(:&552Adam/dense_1002/kernel/m
": 52Adam/dense_1002/bias/m
0:.52$Adam/batch_normalization_905/gamma/m
/:-52#Adam/batch_normalization_905/beta/m
(:&52Adam/dense_1003/kernel/m
": 2Adam/dense_1003/bias/m
0:.2$Adam/batch_normalization_906/gamma/m
/:-2#Adam/batch_normalization_906/beta/m
(:&2Adam/dense_1004/kernel/m
": 2Adam/dense_1004/bias/m
0:.2$Adam/batch_normalization_907/gamma/m
/:-2#Adam/batch_normalization_907/beta/m
(:&2Adam/dense_1005/kernel/m
": 2Adam/dense_1005/bias/m
0:.2$Adam/batch_normalization_908/gamma/m
/:-2#Adam/batch_normalization_908/beta/m
(:&2Adam/dense_1006/kernel/m
": 2Adam/dense_1006/bias/m
0:.2$Adam/batch_normalization_909/gamma/m
/:-2#Adam/batch_normalization_909/beta/m
(:&2Adam/dense_1007/kernel/m
": 2Adam/dense_1007/bias/m
0:.2$Adam/batch_normalization_910/gamma/m
/:-2#Adam/batch_normalization_910/beta/m
(:&2Adam/dense_1008/kernel/m
": 2Adam/dense_1008/bias/m
0:.2$Adam/batch_normalization_911/gamma/m
/:-2#Adam/batch_normalization_911/beta/m
(:&2Adam/dense_1009/kernel/m
": 2Adam/dense_1009/bias/m
0:.2$Adam/batch_normalization_912/gamma/m
/:-2#Adam/batch_normalization_912/beta/m
(:&2Adam/dense_1010/kernel/m
": 2Adam/dense_1010/bias/m
0:.2$Adam/batch_normalization_913/gamma/m
/:-2#Adam/batch_normalization_913/beta/m
(:&2Adam/dense_1011/kernel/m
": 2Adam/dense_1011/bias/m
(:&52Adam/dense_1001/kernel/v
": 52Adam/dense_1001/bias/v
0:.52$Adam/batch_normalization_904/gamma/v
/:-52#Adam/batch_normalization_904/beta/v
(:&552Adam/dense_1002/kernel/v
": 52Adam/dense_1002/bias/v
0:.52$Adam/batch_normalization_905/gamma/v
/:-52#Adam/batch_normalization_905/beta/v
(:&52Adam/dense_1003/kernel/v
": 2Adam/dense_1003/bias/v
0:.2$Adam/batch_normalization_906/gamma/v
/:-2#Adam/batch_normalization_906/beta/v
(:&2Adam/dense_1004/kernel/v
": 2Adam/dense_1004/bias/v
0:.2$Adam/batch_normalization_907/gamma/v
/:-2#Adam/batch_normalization_907/beta/v
(:&2Adam/dense_1005/kernel/v
": 2Adam/dense_1005/bias/v
0:.2$Adam/batch_normalization_908/gamma/v
/:-2#Adam/batch_normalization_908/beta/v
(:&2Adam/dense_1006/kernel/v
": 2Adam/dense_1006/bias/v
0:.2$Adam/batch_normalization_909/gamma/v
/:-2#Adam/batch_normalization_909/beta/v
(:&2Adam/dense_1007/kernel/v
": 2Adam/dense_1007/bias/v
0:.2$Adam/batch_normalization_910/gamma/v
/:-2#Adam/batch_normalization_910/beta/v
(:&2Adam/dense_1008/kernel/v
": 2Adam/dense_1008/bias/v
0:.2$Adam/batch_normalization_911/gamma/v
/:-2#Adam/batch_normalization_911/beta/v
(:&2Adam/dense_1009/kernel/v
": 2Adam/dense_1009/bias/v
0:.2$Adam/batch_normalization_912/gamma/v
/:-2#Adam/batch_normalization_912/beta/v
(:&2Adam/dense_1010/kernel/v
": 2Adam/dense_1010/bias/v
0:.2$Adam/batch_normalization_913/gamma/v
/:-2#Adam/batch_normalization_913/beta/v
(:&2Adam/dense_1011/kernel/v
": 2Adam/dense_1011/bias/v
	J
Const
J	
Const_1
!__inference__wrapped_model_999444èl³´34?<>=LMXUWVefqnpo~£ ¢¡°±¼¹»ºÉÊÕÒÔÓâãîëíìûü ­®?¢<
5¢2
0-
normalization_97_inputÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

dense_1011$!

dense_1011ÿÿÿÿÿÿÿÿÿp
__inference_adapt_step_1002899N0./C¢@
9¢6
41¢
ÿÿÿÿÿÿÿÿÿIteratorSpec 
ª "
 º
T__inference_batch_normalization_904_layer_call_and_return_conditional_losses_1002964b?<>=3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 º
T__inference_batch_normalization_904_layer_call_and_return_conditional_losses_1002998b>?<=3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
9__inference_batch_normalization_904_layer_call_fn_1002931U?<>=3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p 
ª "ÿÿÿÿÿÿÿÿÿ5
9__inference_batch_normalization_904_layer_call_fn_1002944U>?<=3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p
ª "ÿÿÿÿÿÿÿÿÿ5º
T__inference_batch_normalization_905_layer_call_and_return_conditional_losses_1003073bXUWV3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 º
T__inference_batch_normalization_905_layer_call_and_return_conditional_losses_1003107bWXUV3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
9__inference_batch_normalization_905_layer_call_fn_1003040UXUWV3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p 
ª "ÿÿÿÿÿÿÿÿÿ5
9__inference_batch_normalization_905_layer_call_fn_1003053UWXUV3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ5
p
ª "ÿÿÿÿÿÿÿÿÿ5º
T__inference_batch_normalization_906_layer_call_and_return_conditional_losses_1003182bqnpo3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
T__inference_batch_normalization_906_layer_call_and_return_conditional_losses_1003216bpqno3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_906_layer_call_fn_1003149Uqnpo3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_906_layer_call_fn_1003162Upqno3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¾
T__inference_batch_normalization_907_layer_call_and_return_conditional_losses_1003291f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
T__inference_batch_normalization_907_layer_call_and_return_conditional_losses_1003325f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_907_layer_call_fn_1003258Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_907_layer_call_fn_1003271Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¾
T__inference_batch_normalization_908_layer_call_and_return_conditional_losses_1003400f£ ¢¡3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
T__inference_batch_normalization_908_layer_call_and_return_conditional_losses_1003434f¢£ ¡3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_908_layer_call_fn_1003367Y£ ¢¡3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_908_layer_call_fn_1003380Y¢£ ¡3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¾
T__inference_batch_normalization_909_layer_call_and_return_conditional_losses_1003509f¼¹»º3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
T__inference_batch_normalization_909_layer_call_and_return_conditional_losses_1003543f»¼¹º3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_909_layer_call_fn_1003476Y¼¹»º3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_909_layer_call_fn_1003489Y»¼¹º3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¾
T__inference_batch_normalization_910_layer_call_and_return_conditional_losses_1003618fÕÒÔÓ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
T__inference_batch_normalization_910_layer_call_and_return_conditional_losses_1003652fÔÕÒÓ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_910_layer_call_fn_1003585YÕÒÔÓ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_910_layer_call_fn_1003598YÔÕÒÓ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¾
T__inference_batch_normalization_911_layer_call_and_return_conditional_losses_1003727fîëíì3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
T__inference_batch_normalization_911_layer_call_and_return_conditional_losses_1003761fíîëì3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_911_layer_call_fn_1003694Yîëíì3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_911_layer_call_fn_1003707Yíîëì3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¾
T__inference_batch_normalization_912_layer_call_and_return_conditional_losses_1003836f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
T__inference_batch_normalization_912_layer_call_and_return_conditional_losses_1003870f3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_912_layer_call_fn_1003803Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_912_layer_call_fn_1003816Y3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¾
T__inference_batch_normalization_913_layer_call_and_return_conditional_losses_1003945f 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
T__inference_batch_normalization_913_layer_call_and_return_conditional_losses_1003979f 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_913_layer_call_fn_1003912Y 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_913_layer_call_fn_1003925Y 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_dense_1001_layer_call_and_return_conditional_losses_1002918\34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
,__inference_dense_1001_layer_call_fn_1002908O34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ5§
G__inference_dense_1002_layer_call_and_return_conditional_losses_1003027\LM/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
,__inference_dense_1002_layer_call_fn_1003017OLM/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "ÿÿÿÿÿÿÿÿÿ5§
G__inference_dense_1003_layer_call_and_return_conditional_losses_1003136\ef/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_1003_layer_call_fn_1003126Oef/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_dense_1004_layer_call_and_return_conditional_losses_1003245\~/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_1004_layer_call_fn_1003235O~/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_1005_layer_call_and_return_conditional_losses_1003354^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_1005_layer_call_fn_1003344Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_1006_layer_call_and_return_conditional_losses_1003463^°±/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_1006_layer_call_fn_1003453Q°±/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_1007_layer_call_and_return_conditional_losses_1003572^ÉÊ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_1007_layer_call_fn_1003562QÉÊ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_1008_layer_call_and_return_conditional_losses_1003681^âã/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_1008_layer_call_fn_1003671Qâã/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_1009_layer_call_and_return_conditional_losses_1003790^ûü/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_1009_layer_call_fn_1003780Qûü/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_1010_layer_call_and_return_conditional_losses_1003899^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_1010_layer_call_fn_1003889Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_1011_layer_call_and_return_conditional_losses_1004008^­®/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_1011_layer_call_fn_1003998Q­®/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
L__inference_leaky_re_lu_904_layer_call_and_return_conditional_losses_1003008X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
1__inference_leaky_re_lu_904_layer_call_fn_1003003K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "ÿÿÿÿÿÿÿÿÿ5¨
L__inference_leaky_re_lu_905_layer_call_and_return_conditional_losses_1003117X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ5
 
1__inference_leaky_re_lu_905_layer_call_fn_1003112K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ5
ª "ÿÿÿÿÿÿÿÿÿ5¨
L__inference_leaky_re_lu_906_layer_call_and_return_conditional_losses_1003226X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_leaky_re_lu_906_layer_call_fn_1003221K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
L__inference_leaky_re_lu_907_layer_call_and_return_conditional_losses_1003335X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_leaky_re_lu_907_layer_call_fn_1003330K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
L__inference_leaky_re_lu_908_layer_call_and_return_conditional_losses_1003444X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_leaky_re_lu_908_layer_call_fn_1003439K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
L__inference_leaky_re_lu_909_layer_call_and_return_conditional_losses_1003553X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_leaky_re_lu_909_layer_call_fn_1003548K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
L__inference_leaky_re_lu_910_layer_call_and_return_conditional_losses_1003662X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_leaky_re_lu_910_layer_call_fn_1003657K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
L__inference_leaky_re_lu_911_layer_call_and_return_conditional_losses_1003771X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_leaky_re_lu_911_layer_call_fn_1003766K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
L__inference_leaky_re_lu_912_layer_call_and_return_conditional_losses_1003880X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_leaky_re_lu_912_layer_call_fn_1003875K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
L__inference_leaky_re_lu_913_layer_call_and_return_conditional_losses_1003989X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_leaky_re_lu_913_layer_call_fn_1003984K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ­
J__inference_sequential_97_layer_call_and_return_conditional_losses_1001647Þl³´34?<>=LMXUWVefqnpo~£ ¢¡°±¼¹»ºÉÊÕÒÔÓâãîëíìûü ­®G¢D
=¢:
0-
normalization_97_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ­
J__inference_sequential_97_layer_call_and_return_conditional_losses_1001813Þl³´34>?<=LMWXUVefpqno~¢£ ¡°±»¼¹ºÉÊÔÕÒÓâãíîëìûü ­®G¢D
=¢:
0-
normalization_97_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
J__inference_sequential_97_layer_call_and_return_conditional_losses_1002330Îl³´34?<>=LMXUWVefqnpo~£ ¢¡°±¼¹»ºÉÊÕÒÔÓâãîëíìûü ­®7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
J__inference_sequential_97_layer_call_and_return_conditional_losses_1002717Îl³´34>?<=LMWXUVefpqno~¢£ ¡°±»¼¹ºÉÊÔÕÒÓâãíîëìûü ­®7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_sequential_97_layer_call_fn_1000746Ñl³´34?<>=LMXUWVefqnpo~£ ¢¡°±¼¹»ºÉÊÕÒÔÓâãîëíìûü ­®G¢D
=¢:
0-
normalization_97_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_97_layer_call_fn_1001481Ñl³´34>?<=LMWXUVefpqno~¢£ ¡°±»¼¹ºÉÊÔÕÒÓâãíîëìûü ­®G¢D
=¢:
0-
normalization_97_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿõ
/__inference_sequential_97_layer_call_fn_1001950Ál³´34?<>=LMXUWVefqnpo~£ ¢¡°±¼¹»ºÉÊÕÒÔÓâãîëíìûü ­®7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿõ
/__inference_sequential_97_layer_call_fn_1002083Ál³´34>?<=LMWXUVefpqno~¢£ ¡°±»¼¹ºÉÊÔÕÒÓâãíîëìûü ­®7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¬
%__inference_signature_wrapper_1002852l³´34?<>=LMXUWVefqnpo~£ ¢¡°±¼¹»ºÉÊÕÒÔÓâãîëíìûü ­®Y¢V
¢ 
OªL
J
normalization_97_input0-
normalization_97_inputÿÿÿÿÿÿÿÿÿ"7ª4
2

dense_1011$!

dense_1011ÿÿÿÿÿÿÿÿÿ