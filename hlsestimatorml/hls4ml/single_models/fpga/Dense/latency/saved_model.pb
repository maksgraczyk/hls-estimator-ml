ó
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Ê¡
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
dense_695/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_695/kernel
u
$dense_695/kernel/Read/ReadVariableOpReadVariableOpdense_695/kernel*
_output_shapes

:*
dtype0
t
dense_695/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_695/bias
m
"dense_695/bias/Read/ReadVariableOpReadVariableOpdense_695/bias*
_output_shapes
:*
dtype0

batch_normalization_625/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_625/gamma

1batch_normalization_625/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_625/gamma*
_output_shapes
:*
dtype0

batch_normalization_625/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_625/beta

0batch_normalization_625/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_625/beta*
_output_shapes
:*
dtype0

#batch_normalization_625/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_625/moving_mean

7batch_normalization_625/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_625/moving_mean*
_output_shapes
:*
dtype0
¦
'batch_normalization_625/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_625/moving_variance

;batch_normalization_625/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_625/moving_variance*
_output_shapes
:*
dtype0
|
dense_696/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7*!
shared_namedense_696/kernel
u
$dense_696/kernel/Read/ReadVariableOpReadVariableOpdense_696/kernel*
_output_shapes

:7*
dtype0
t
dense_696/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*
shared_namedense_696/bias
m
"dense_696/bias/Read/ReadVariableOpReadVariableOpdense_696/bias*
_output_shapes
:7*
dtype0

batch_normalization_626/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*.
shared_namebatch_normalization_626/gamma

1batch_normalization_626/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_626/gamma*
_output_shapes
:7*
dtype0

batch_normalization_626/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*-
shared_namebatch_normalization_626/beta

0batch_normalization_626/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_626/beta*
_output_shapes
:7*
dtype0

#batch_normalization_626/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*4
shared_name%#batch_normalization_626/moving_mean

7batch_normalization_626/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_626/moving_mean*
_output_shapes
:7*
dtype0
¦
'batch_normalization_626/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*8
shared_name)'batch_normalization_626/moving_variance

;batch_normalization_626/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_626/moving_variance*
_output_shapes
:7*
dtype0
|
dense_697/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7F*!
shared_namedense_697/kernel
u
$dense_697/kernel/Read/ReadVariableOpReadVariableOpdense_697/kernel*
_output_shapes

:7F*
dtype0
t
dense_697/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*
shared_namedense_697/bias
m
"dense_697/bias/Read/ReadVariableOpReadVariableOpdense_697/bias*
_output_shapes
:F*
dtype0

batch_normalization_627/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*.
shared_namebatch_normalization_627/gamma

1batch_normalization_627/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_627/gamma*
_output_shapes
:F*
dtype0

batch_normalization_627/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*-
shared_namebatch_normalization_627/beta

0batch_normalization_627/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_627/beta*
_output_shapes
:F*
dtype0

#batch_normalization_627/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*4
shared_name%#batch_normalization_627/moving_mean

7batch_normalization_627/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_627/moving_mean*
_output_shapes
:F*
dtype0
¦
'batch_normalization_627/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*8
shared_name)'batch_normalization_627/moving_variance

;batch_normalization_627/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_627/moving_variance*
_output_shapes
:F*
dtype0
|
dense_698/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*!
shared_namedense_698/kernel
u
$dense_698/kernel/Read/ReadVariableOpReadVariableOpdense_698/kernel*
_output_shapes

:F*
dtype0
t
dense_698/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_698/bias
m
"dense_698/bias/Read/ReadVariableOpReadVariableOpdense_698/bias*
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
Adam/dense_695/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_695/kernel/m

+Adam/dense_695/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_695/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_695/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_695/bias/m
{
)Adam/dense_695/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_695/bias/m*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_625/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_625/gamma/m

8Adam/batch_normalization_625/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_625/gamma/m*
_output_shapes
:*
dtype0

#Adam/batch_normalization_625/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_625/beta/m

7Adam/batch_normalization_625/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_625/beta/m*
_output_shapes
:*
dtype0

Adam/dense_696/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7*(
shared_nameAdam/dense_696/kernel/m

+Adam/dense_696/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_696/kernel/m*
_output_shapes

:7*
dtype0

Adam/dense_696/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*&
shared_nameAdam/dense_696/bias/m
{
)Adam/dense_696/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_696/bias/m*
_output_shapes
:7*
dtype0
 
$Adam/batch_normalization_626/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*5
shared_name&$Adam/batch_normalization_626/gamma/m

8Adam/batch_normalization_626/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_626/gamma/m*
_output_shapes
:7*
dtype0

#Adam/batch_normalization_626/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*4
shared_name%#Adam/batch_normalization_626/beta/m

7Adam/batch_normalization_626/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_626/beta/m*
_output_shapes
:7*
dtype0

Adam/dense_697/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7F*(
shared_nameAdam/dense_697/kernel/m

+Adam/dense_697/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_697/kernel/m*
_output_shapes

:7F*
dtype0

Adam/dense_697/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*&
shared_nameAdam/dense_697/bias/m
{
)Adam/dense_697/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_697/bias/m*
_output_shapes
:F*
dtype0
 
$Adam/batch_normalization_627/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*5
shared_name&$Adam/batch_normalization_627/gamma/m

8Adam/batch_normalization_627/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_627/gamma/m*
_output_shapes
:F*
dtype0

#Adam/batch_normalization_627/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*4
shared_name%#Adam/batch_normalization_627/beta/m

7Adam/batch_normalization_627/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_627/beta/m*
_output_shapes
:F*
dtype0

Adam/dense_698/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*(
shared_nameAdam/dense_698/kernel/m

+Adam/dense_698/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_698/kernel/m*
_output_shapes

:F*
dtype0

Adam/dense_698/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_698/bias/m
{
)Adam/dense_698/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_698/bias/m*
_output_shapes
:*
dtype0

Adam/dense_695/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_695/kernel/v

+Adam/dense_695/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_695/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_695/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_695/bias/v
{
)Adam/dense_695/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_695/bias/v*
_output_shapes
:*
dtype0
 
$Adam/batch_normalization_625/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_625/gamma/v

8Adam/batch_normalization_625/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_625/gamma/v*
_output_shapes
:*
dtype0

#Adam/batch_normalization_625/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_625/beta/v

7Adam/batch_normalization_625/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_625/beta/v*
_output_shapes
:*
dtype0

Adam/dense_696/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7*(
shared_nameAdam/dense_696/kernel/v

+Adam/dense_696/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_696/kernel/v*
_output_shapes

:7*
dtype0

Adam/dense_696/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*&
shared_nameAdam/dense_696/bias/v
{
)Adam/dense_696/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_696/bias/v*
_output_shapes
:7*
dtype0
 
$Adam/batch_normalization_626/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*5
shared_name&$Adam/batch_normalization_626/gamma/v

8Adam/batch_normalization_626/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_626/gamma/v*
_output_shapes
:7*
dtype0

#Adam/batch_normalization_626/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*4
shared_name%#Adam/batch_normalization_626/beta/v

7Adam/batch_normalization_626/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_626/beta/v*
_output_shapes
:7*
dtype0

Adam/dense_697/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:7F*(
shared_nameAdam/dense_697/kernel/v

+Adam/dense_697/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_697/kernel/v*
_output_shapes

:7F*
dtype0

Adam/dense_697/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*&
shared_nameAdam/dense_697/bias/v
{
)Adam/dense_697/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_697/bias/v*
_output_shapes
:F*
dtype0
 
$Adam/batch_normalization_627/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*5
shared_name&$Adam/batch_normalization_627/gamma/v

8Adam/batch_normalization_627/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_627/gamma/v*
_output_shapes
:F*
dtype0

#Adam/batch_normalization_627/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*4
shared_name%#Adam/batch_normalization_627/beta/v

7Adam/batch_normalization_627/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_627/beta/v*
_output_shapes
:F*
dtype0

Adam/dense_698/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*(
shared_nameAdam/dense_698/kernel/v

+Adam/dense_698/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_698/kernel/v*
_output_shapes

:F*
dtype0

Adam/dense_698/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_698/bias/v
{
)Adam/dense_698/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_698/bias/v*
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
Øo
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*o
valueoBo Býn
¬
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
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¾

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
	keras_api
_adapt_function*
¦

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
Õ
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*

1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses* 
¦

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
Õ
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses*

J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
¦

Pkernel
Qbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses*
Õ
Xaxis
	Ygamma
Zbeta
[moving_mean
\moving_variance
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses*

c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses* 
¦

ikernel
jbias
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses*
É
qiter

rbeta_1

sbeta_2
	tdecaymµm¶'m·(m¸7m¹8mº@m»Am¼Pm½Qm¾Ym¿ZmÀimÁjmÂvÃvÄ'vÅ(vÆ7vÇ8vÈ@vÉAvÊPvËQvÌYvÍZvÎivÏjvÐ*
²
0
1
2
3
4
'5
(6
)7
*8
79
810
@11
A12
B13
C14
P15
Q16
Y17
Z18
[19
\20
i21
j22*
j
0
1
'2
(3
74
85
@6
A7
P8
Q9
Y10
Z11
i12
j13*

u0
v1
w2* 
°
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

}serving_default* 
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
VARIABLE_VALUEdense_695/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_695/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
	
u0* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_625/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_625/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_625/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_625/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
'0
(1
)2
*3*

'0
(1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_696/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_696/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*
	
v0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_626/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_626/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_626/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_626/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
@0
A1
B2
C3*

@0
A1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_697/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_697/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

P0
Q1*

P0
Q1*
	
w0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_627/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_627/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_627/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE'batch_normalization_627/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
Y0
Z1
[2
\3*

Y0
Z1*
* 

¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_698/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_698/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

i0
j1*

i0
j1*
* 

«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*
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
C
0
1
2
)3
*4
B5
C6
[7
\8*
R
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
10*

°0*
* 
* 
* 
* 
* 
* 
	
u0* 
* 

)0
*1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
v0* 
* 

B0
C1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
w0* 
* 

[0
\1*
* 
* 
* 
* 
* 
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

±total

²count
³	variables
´	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

±0
²1*

³	variables*
}
VARIABLE_VALUEAdam/dense_695/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_695/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_625/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_625/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_696/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_696/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_626/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_626/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_697/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_697/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_627/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_627/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_698/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_698/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_695/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_695/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_625/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_625/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_696/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_696/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_626/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_626/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_697/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_697/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE$Adam/batch_normalization_627/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_627/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_698/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_698/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

&serving_default_normalization_70_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
¥
StatefulPartitionedCallStatefulPartitionedCall&serving_default_normalization_70_inputConstConst_1dense_695/kerneldense_695/bias'batch_normalization_625/moving_variancebatch_normalization_625/gamma#batch_normalization_625/moving_meanbatch_normalization_625/betadense_696/kerneldense_696/bias'batch_normalization_626/moving_variancebatch_normalization_626/gamma#batch_normalization_626/moving_meanbatch_normalization_626/betadense_697/kerneldense_697/bias'batch_normalization_627/moving_variancebatch_normalization_627/gamma#batch_normalization_627/moving_meanbatch_normalization_627/betadense_698/kerneldense_698/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1237562
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp$dense_695/kernel/Read/ReadVariableOp"dense_695/bias/Read/ReadVariableOp1batch_normalization_625/gamma/Read/ReadVariableOp0batch_normalization_625/beta/Read/ReadVariableOp7batch_normalization_625/moving_mean/Read/ReadVariableOp;batch_normalization_625/moving_variance/Read/ReadVariableOp$dense_696/kernel/Read/ReadVariableOp"dense_696/bias/Read/ReadVariableOp1batch_normalization_626/gamma/Read/ReadVariableOp0batch_normalization_626/beta/Read/ReadVariableOp7batch_normalization_626/moving_mean/Read/ReadVariableOp;batch_normalization_626/moving_variance/Read/ReadVariableOp$dense_697/kernel/Read/ReadVariableOp"dense_697/bias/Read/ReadVariableOp1batch_normalization_627/gamma/Read/ReadVariableOp0batch_normalization_627/beta/Read/ReadVariableOp7batch_normalization_627/moving_mean/Read/ReadVariableOp;batch_normalization_627/moving_variance/Read/ReadVariableOp$dense_698/kernel/Read/ReadVariableOp"dense_698/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_695/kernel/m/Read/ReadVariableOp)Adam/dense_695/bias/m/Read/ReadVariableOp8Adam/batch_normalization_625/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_625/beta/m/Read/ReadVariableOp+Adam/dense_696/kernel/m/Read/ReadVariableOp)Adam/dense_696/bias/m/Read/ReadVariableOp8Adam/batch_normalization_626/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_626/beta/m/Read/ReadVariableOp+Adam/dense_697/kernel/m/Read/ReadVariableOp)Adam/dense_697/bias/m/Read/ReadVariableOp8Adam/batch_normalization_627/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_627/beta/m/Read/ReadVariableOp+Adam/dense_698/kernel/m/Read/ReadVariableOp)Adam/dense_698/bias/m/Read/ReadVariableOp+Adam/dense_695/kernel/v/Read/ReadVariableOp)Adam/dense_695/bias/v/Read/ReadVariableOp8Adam/batch_normalization_625/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_625/beta/v/Read/ReadVariableOp+Adam/dense_696/kernel/v/Read/ReadVariableOp)Adam/dense_696/bias/v/Read/ReadVariableOp8Adam/batch_normalization_626/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_626/beta/v/Read/ReadVariableOp+Adam/dense_697/kernel/v/Read/ReadVariableOp)Adam/dense_697/bias/v/Read/ReadVariableOp8Adam/batch_normalization_627/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_627/beta/v/Read/ReadVariableOp+Adam/dense_698/kernel/v/Read/ReadVariableOp)Adam/dense_698/bias/v/Read/ReadVariableOpConst_2*F
Tin?
=2;		*
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
 __inference__traced_save_1238220

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_695/kerneldense_695/biasbatch_normalization_625/gammabatch_normalization_625/beta#batch_normalization_625/moving_mean'batch_normalization_625/moving_variancedense_696/kerneldense_696/biasbatch_normalization_626/gammabatch_normalization_626/beta#batch_normalization_626/moving_mean'batch_normalization_626/moving_variancedense_697/kerneldense_697/biasbatch_normalization_627/gammabatch_normalization_627/beta#batch_normalization_627/moving_mean'batch_normalization_627/moving_variancedense_698/kerneldense_698/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount_1Adam/dense_695/kernel/mAdam/dense_695/bias/m$Adam/batch_normalization_625/gamma/m#Adam/batch_normalization_625/beta/mAdam/dense_696/kernel/mAdam/dense_696/bias/m$Adam/batch_normalization_626/gamma/m#Adam/batch_normalization_626/beta/mAdam/dense_697/kernel/mAdam/dense_697/bias/m$Adam/batch_normalization_627/gamma/m#Adam/batch_normalization_627/beta/mAdam/dense_698/kernel/mAdam/dense_698/bias/mAdam/dense_695/kernel/vAdam/dense_695/bias/v$Adam/batch_normalization_625/gamma/v#Adam/batch_normalization_625/beta/vAdam/dense_696/kernel/vAdam/dense_696/bias/v$Adam/batch_normalization_626/gamma/v#Adam/batch_normalization_626/beta/vAdam/dense_697/kernel/vAdam/dense_697/bias/v$Adam/batch_normalization_627/gamma/v#Adam/batch_normalization_627/beta/vAdam/dense_698/kernel/vAdam/dense_698/bias/v*E
Tin>
<2:*
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
#__inference__traced_restore_1238401
©
®
__inference_loss_fn_2_1238024J
8dense_697_kernel_regularizer_abs_readvariableop_resource:7F
identity¢/dense_697/kernel/Regularizer/Abs/ReadVariableOp¨
/dense_697/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8dense_697_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:7F*
dtype0
 dense_697/kernel/Regularizer/AbsAbs7dense_697/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7Fs
"dense_697/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_697/kernel/Regularizer/SumSum$dense_697/kernel/Regularizer/Abs:y:0+dense_697/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_697/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 dense_697/kernel/Regularizer/mulMul+dense_697/kernel/Regularizer/mul/x:output:0)dense_697/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_697/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^dense_697/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_697/kernel/Regularizer/Abs/ReadVariableOp/dense_697/kernel/Regularizer/Abs/ReadVariableOp
æ
h
L__inference_leaky_re_lu_626_layer_call_and_return_conditional_losses_1237851

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
É	
÷
F__inference_dense_698_layer_call_and_return_conditional_losses_1237991

inputs0
matmul_readvariableop_resource:F-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
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
:ÿÿÿÿÿÿÿÿÿF: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
t

 __inference__traced_save_1238220
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	/
+savev2_dense_695_kernel_read_readvariableop-
)savev2_dense_695_bias_read_readvariableop<
8savev2_batch_normalization_625_gamma_read_readvariableop;
7savev2_batch_normalization_625_beta_read_readvariableopB
>savev2_batch_normalization_625_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_625_moving_variance_read_readvariableop/
+savev2_dense_696_kernel_read_readvariableop-
)savev2_dense_696_bias_read_readvariableop<
8savev2_batch_normalization_626_gamma_read_readvariableop;
7savev2_batch_normalization_626_beta_read_readvariableopB
>savev2_batch_normalization_626_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_626_moving_variance_read_readvariableop/
+savev2_dense_697_kernel_read_readvariableop-
)savev2_dense_697_bias_read_readvariableop<
8savev2_batch_normalization_627_gamma_read_readvariableop;
7savev2_batch_normalization_627_beta_read_readvariableopB
>savev2_batch_normalization_627_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_627_moving_variance_read_readvariableop/
+savev2_dense_698_kernel_read_readvariableop-
)savev2_dense_698_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_695_kernel_m_read_readvariableop4
0savev2_adam_dense_695_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_625_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_625_beta_m_read_readvariableop6
2savev2_adam_dense_696_kernel_m_read_readvariableop4
0savev2_adam_dense_696_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_626_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_626_beta_m_read_readvariableop6
2savev2_adam_dense_697_kernel_m_read_readvariableop4
0savev2_adam_dense_697_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_627_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_627_beta_m_read_readvariableop6
2savev2_adam_dense_698_kernel_m_read_readvariableop4
0savev2_adam_dense_698_bias_m_read_readvariableop6
2savev2_adam_dense_695_kernel_v_read_readvariableop4
0savev2_adam_dense_695_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_625_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_625_beta_v_read_readvariableop6
2savev2_adam_dense_696_kernel_v_read_readvariableop4
0savev2_adam_dense_696_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_626_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_626_beta_v_read_readvariableop6
2savev2_adam_dense_697_kernel_v_read_readvariableop4
0savev2_adam_dense_697_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_627_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_627_beta_v_read_readvariableop6
2savev2_adam_dense_698_kernel_v_read_readvariableop4
0savev2_adam_dense_698_bias_v_read_readvariableop
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
: Õ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*þ
valueôBñ:B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHâ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ±
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop+savev2_dense_695_kernel_read_readvariableop)savev2_dense_695_bias_read_readvariableop8savev2_batch_normalization_625_gamma_read_readvariableop7savev2_batch_normalization_625_beta_read_readvariableop>savev2_batch_normalization_625_moving_mean_read_readvariableopBsavev2_batch_normalization_625_moving_variance_read_readvariableop+savev2_dense_696_kernel_read_readvariableop)savev2_dense_696_bias_read_readvariableop8savev2_batch_normalization_626_gamma_read_readvariableop7savev2_batch_normalization_626_beta_read_readvariableop>savev2_batch_normalization_626_moving_mean_read_readvariableopBsavev2_batch_normalization_626_moving_variance_read_readvariableop+savev2_dense_697_kernel_read_readvariableop)savev2_dense_697_bias_read_readvariableop8savev2_batch_normalization_627_gamma_read_readvariableop7savev2_batch_normalization_627_beta_read_readvariableop>savev2_batch_normalization_627_moving_mean_read_readvariableopBsavev2_batch_normalization_627_moving_variance_read_readvariableop+savev2_dense_698_kernel_read_readvariableop)savev2_dense_698_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_695_kernel_m_read_readvariableop0savev2_adam_dense_695_bias_m_read_readvariableop?savev2_adam_batch_normalization_625_gamma_m_read_readvariableop>savev2_adam_batch_normalization_625_beta_m_read_readvariableop2savev2_adam_dense_696_kernel_m_read_readvariableop0savev2_adam_dense_696_bias_m_read_readvariableop?savev2_adam_batch_normalization_626_gamma_m_read_readvariableop>savev2_adam_batch_normalization_626_beta_m_read_readvariableop2savev2_adam_dense_697_kernel_m_read_readvariableop0savev2_adam_dense_697_bias_m_read_readvariableop?savev2_adam_batch_normalization_627_gamma_m_read_readvariableop>savev2_adam_batch_normalization_627_beta_m_read_readvariableop2savev2_adam_dense_698_kernel_m_read_readvariableop0savev2_adam_dense_698_bias_m_read_readvariableop2savev2_adam_dense_695_kernel_v_read_readvariableop0savev2_adam_dense_695_bias_v_read_readvariableop?savev2_adam_batch_normalization_625_gamma_v_read_readvariableop>savev2_adam_batch_normalization_625_beta_v_read_readvariableop2savev2_adam_dense_696_kernel_v_read_readvariableop0savev2_adam_dense_696_bias_v_read_readvariableop?savev2_adam_batch_normalization_626_gamma_v_read_readvariableop>savev2_adam_batch_normalization_626_beta_v_read_readvariableop2savev2_adam_dense_697_kernel_v_read_readvariableop0savev2_adam_dense_697_bias_v_read_readvariableop?savev2_adam_batch_normalization_627_gamma_v_read_readvariableop>savev2_adam_batch_normalization_627_beta_v_read_readvariableop2savev2_adam_dense_698_kernel_v_read_readvariableop0savev2_adam_dense_698_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:		
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

identity_1Identity_1:output:0*
_input_shapesñ
î: ::: :::::::7:7:7:7:7:7:7F:F:F:F:F:F:F:: : : : : : :::::7:7:7:7:7F:F:F:F:F::::::7:7:7:7:7F:F:F:F:F:: 2(
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

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
::$
 

_output_shapes

:7: 
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

:7F: 

_output_shapes
:F: 

_output_shapes
:F: 

_output_shapes
:F: 

_output_shapes
:F: 

_output_shapes
:F:$ 

_output_shapes

:F: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
::$" 

_output_shapes

:7: #

_output_shapes
:7: $

_output_shapes
:7: %

_output_shapes
:7:$& 

_output_shapes

:7F: '

_output_shapes
:F: (

_output_shapes
:F: )

_output_shapes
:F:$* 

_output_shapes

:F: +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
::$0 

_output_shapes

:7: 1
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

:7F: 5

_output_shapes
:F: 6

_output_shapes
:F: 7

_output_shapes
:F:$8 

_output_shapes

:F: 9

_output_shapes
:::

_output_shapes
: 

£
/__inference_sequential_70_layer_call_fn_1236699
normalization_70_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:7
	unknown_8:7
	unknown_9:7

unknown_10:7

unknown_11:7

unknown_12:7

unknown_13:7F

unknown_14:F

unknown_15:F

unknown_16:F

unknown_17:F

unknown_18:F

unknown_19:F

unknown_20:
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallnormalization_70_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_70_layer_call_and_return_conditional_losses_1236652o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_70_input:$ 

_output_shapes

::$ 

_output_shapes

:
%
í
T__inference_batch_normalization_625_layer_call_and_return_conditional_losses_1236314

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_625_layer_call_fn_1237725

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_625_layer_call_and_return_conditional_losses_1236539`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_627_layer_call_and_return_conditional_losses_1237972

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Æ

+__inference_dense_697_layer_call_fn_1237866

inputs
unknown:7F
	unknown_0:F
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_697_layer_call_and_return_conditional_losses_1236595o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF`
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
®
Ô
9__inference_batch_normalization_625_layer_call_fn_1237653

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_625_layer_call_and_return_conditional_losses_1236267o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

+__inference_dense_695_layer_call_fn_1237624

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_695_layer_call_and_return_conditional_losses_1236519o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
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
®
Ô
9__inference_batch_normalization_627_layer_call_fn_1237895

inputs
unknown:F
	unknown_0:F
	unknown_1:F
	unknown_2:F
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_627_layer_call_and_return_conditional_losses_1236431o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
¬
Ô
9__inference_batch_normalization_627_layer_call_fn_1237908

inputs
unknown:F
	unknown_0:F
	unknown_1:F
	unknown_2:F
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_627_layer_call_and_return_conditional_losses_1236478o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
©
®
__inference_loss_fn_1_1238013J
8dense_696_kernel_regularizer_abs_readvariableop_resource:7
identity¢/dense_696/kernel/Regularizer/Abs/ReadVariableOp¨
/dense_696/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8dense_696_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:7*
dtype0
 dense_696/kernel/Regularizer/AbsAbs7dense_696/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7s
"dense_696/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_696/kernel/Regularizer/SumSum$dense_696/kernel/Regularizer/Abs:y:0+dense_696/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_696/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_696/kernel/Regularizer/mulMul+dense_696/kernel/Regularizer/mul/x:output:0)dense_696/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_696/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^dense_696/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_696/kernel/Regularizer/Abs/ReadVariableOp/dense_696/kernel/Regularizer/Abs/ReadVariableOp
Î
©
F__inference_dense_696_layer_call_and_return_conditional_losses_1237761

inputs0
matmul_readvariableop_resource:7-
biasadd_readvariableop_resource:7
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_696/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7*
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
:ÿÿÿÿÿÿÿÿÿ7
/dense_696/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7*
dtype0
 dense_696/kernel/Regularizer/AbsAbs7dense_696/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7s
"dense_696/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_696/kernel/Regularizer/SumSum$dense_696/kernel/Regularizer/Abs:y:0+dense_696/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_696/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_696/kernel/Regularizer/mulMul+dense_696/kernel/Regularizer/mul/x:output:0)dense_696/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7©
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_696/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_696/kernel/Regularizer/Abs/ReadVariableOp/dense_696/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

 
J__inference_sequential_70_layer_call_and_return_conditional_losses_1237365

inputs
normalization_70_sub_y
normalization_70_sqrt_x:
(dense_695_matmul_readvariableop_resource:7
)dense_695_biasadd_readvariableop_resource:G
9batch_normalization_625_batchnorm_readvariableop_resource:K
=batch_normalization_625_batchnorm_mul_readvariableop_resource:I
;batch_normalization_625_batchnorm_readvariableop_1_resource:I
;batch_normalization_625_batchnorm_readvariableop_2_resource::
(dense_696_matmul_readvariableop_resource:77
)dense_696_biasadd_readvariableop_resource:7G
9batch_normalization_626_batchnorm_readvariableop_resource:7K
=batch_normalization_626_batchnorm_mul_readvariableop_resource:7I
;batch_normalization_626_batchnorm_readvariableop_1_resource:7I
;batch_normalization_626_batchnorm_readvariableop_2_resource:7:
(dense_697_matmul_readvariableop_resource:7F7
)dense_697_biasadd_readvariableop_resource:FG
9batch_normalization_627_batchnorm_readvariableop_resource:FK
=batch_normalization_627_batchnorm_mul_readvariableop_resource:FI
;batch_normalization_627_batchnorm_readvariableop_1_resource:FI
;batch_normalization_627_batchnorm_readvariableop_2_resource:F:
(dense_698_matmul_readvariableop_resource:F7
)dense_698_biasadd_readvariableop_resource:
identity¢0batch_normalization_625/batchnorm/ReadVariableOp¢2batch_normalization_625/batchnorm/ReadVariableOp_1¢2batch_normalization_625/batchnorm/ReadVariableOp_2¢4batch_normalization_625/batchnorm/mul/ReadVariableOp¢0batch_normalization_626/batchnorm/ReadVariableOp¢2batch_normalization_626/batchnorm/ReadVariableOp_1¢2batch_normalization_626/batchnorm/ReadVariableOp_2¢4batch_normalization_626/batchnorm/mul/ReadVariableOp¢0batch_normalization_627/batchnorm/ReadVariableOp¢2batch_normalization_627/batchnorm/ReadVariableOp_1¢2batch_normalization_627/batchnorm/ReadVariableOp_2¢4batch_normalization_627/batchnorm/mul/ReadVariableOp¢ dense_695/BiasAdd/ReadVariableOp¢dense_695/MatMul/ReadVariableOp¢/dense_695/kernel/Regularizer/Abs/ReadVariableOp¢ dense_696/BiasAdd/ReadVariableOp¢dense_696/MatMul/ReadVariableOp¢/dense_696/kernel/Regularizer/Abs/ReadVariableOp¢ dense_697/BiasAdd/ReadVariableOp¢dense_697/MatMul/ReadVariableOp¢/dense_697/kernel/Regularizer/Abs/ReadVariableOp¢ dense_698/BiasAdd/ReadVariableOp¢dense_698/MatMul/ReadVariableOpm
normalization_70/subSubinputsnormalization_70_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_70/SqrtSqrtnormalization_70_sqrt_x*
T0*
_output_shapes

:_
normalization_70/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_70/MaximumMaximumnormalization_70/Sqrt:y:0#normalization_70/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_70/truedivRealDivnormalization_70/sub:z:0normalization_70/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_695/MatMul/ReadVariableOpReadVariableOp(dense_695_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_695/MatMulMatMulnormalization_70/truediv:z:0'dense_695/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_695/BiasAdd/ReadVariableOpReadVariableOp)dense_695_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_695/BiasAddBiasAdddense_695/MatMul:product:0(dense_695/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0batch_normalization_625/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_625_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'batch_normalization_625/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_625/batchnorm/addAddV28batch_normalization_625/batchnorm/ReadVariableOp:value:00batch_normalization_625/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_625/batchnorm/RsqrtRsqrt)batch_normalization_625/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_625/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_625_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_625/batchnorm/mulMul+batch_normalization_625/batchnorm/Rsqrt:y:0<batch_normalization_625/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_625/batchnorm/mul_1Muldense_695/BiasAdd:output:0)batch_normalization_625/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2batch_normalization_625/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_625_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0º
'batch_normalization_625/batchnorm/mul_2Mul:batch_normalization_625/batchnorm/ReadVariableOp_1:value:0)batch_normalization_625/batchnorm/mul:z:0*
T0*
_output_shapes
:ª
2batch_normalization_625/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_625_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0º
%batch_normalization_625/batchnorm/subSub:batch_normalization_625/batchnorm/ReadVariableOp_2:value:0+batch_normalization_625/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_625/batchnorm/add_1AddV2+batch_normalization_625/batchnorm/mul_1:z:0)batch_normalization_625/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_625/LeakyRelu	LeakyRelu+batch_normalization_625/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_696/MatMul/ReadVariableOpReadVariableOp(dense_696_matmul_readvariableop_resource*
_output_shapes

:7*
dtype0
dense_696/MatMulMatMul'leaky_re_lu_625/LeakyRelu:activations:0'dense_696/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 dense_696/BiasAdd/ReadVariableOpReadVariableOp)dense_696_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0
dense_696/BiasAddBiasAdddense_696/MatMul:product:0(dense_696/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¦
0batch_normalization_626/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_626_batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0l
'batch_normalization_626/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_626/batchnorm/addAddV28batch_normalization_626/batchnorm/ReadVariableOp:value:00batch_normalization_626/batchnorm/add/y:output:0*
T0*
_output_shapes
:7
'batch_normalization_626/batchnorm/RsqrtRsqrt)batch_normalization_626/batchnorm/add:z:0*
T0*
_output_shapes
:7®
4batch_normalization_626/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_626_batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0¼
%batch_normalization_626/batchnorm/mulMul+batch_normalization_626/batchnorm/Rsqrt:y:0<batch_normalization_626/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7§
'batch_normalization_626/batchnorm/mul_1Muldense_696/BiasAdd:output:0)batch_normalization_626/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7ª
2batch_normalization_626/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_626_batchnorm_readvariableop_1_resource*
_output_shapes
:7*
dtype0º
'batch_normalization_626/batchnorm/mul_2Mul:batch_normalization_626/batchnorm/ReadVariableOp_1:value:0)batch_normalization_626/batchnorm/mul:z:0*
T0*
_output_shapes
:7ª
2batch_normalization_626/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_626_batchnorm_readvariableop_2_resource*
_output_shapes
:7*
dtype0º
%batch_normalization_626/batchnorm/subSub:batch_normalization_626/batchnorm/ReadVariableOp_2:value:0+batch_normalization_626/batchnorm/mul_2:z:0*
T0*
_output_shapes
:7º
'batch_normalization_626/batchnorm/add_1AddV2+batch_normalization_626/batchnorm/mul_1:z:0)batch_normalization_626/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
leaky_re_lu_626/LeakyRelu	LeakyRelu+batch_normalization_626/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>
dense_697/MatMul/ReadVariableOpReadVariableOp(dense_697_matmul_readvariableop_resource*
_output_shapes

:7F*
dtype0
dense_697/MatMulMatMul'leaky_re_lu_626/LeakyRelu:activations:0'dense_697/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 dense_697/BiasAdd/ReadVariableOpReadVariableOp)dense_697_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype0
dense_697/BiasAddBiasAdddense_697/MatMul:product:0(dense_697/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF¦
0batch_normalization_627/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_627_batchnorm_readvariableop_resource*
_output_shapes
:F*
dtype0l
'batch_normalization_627/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¿
%batch_normalization_627/batchnorm/addAddV28batch_normalization_627/batchnorm/ReadVariableOp:value:00batch_normalization_627/batchnorm/add/y:output:0*
T0*
_output_shapes
:F
'batch_normalization_627/batchnorm/RsqrtRsqrt)batch_normalization_627/batchnorm/add:z:0*
T0*
_output_shapes
:F®
4batch_normalization_627/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_627_batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype0¼
%batch_normalization_627/batchnorm/mulMul+batch_normalization_627/batchnorm/Rsqrt:y:0<batch_normalization_627/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:F§
'batch_normalization_627/batchnorm/mul_1Muldense_697/BiasAdd:output:0)batch_normalization_627/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFª
2batch_normalization_627/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_627_batchnorm_readvariableop_1_resource*
_output_shapes
:F*
dtype0º
'batch_normalization_627/batchnorm/mul_2Mul:batch_normalization_627/batchnorm/ReadVariableOp_1:value:0)batch_normalization_627/batchnorm/mul:z:0*
T0*
_output_shapes
:Fª
2batch_normalization_627/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_627_batchnorm_readvariableop_2_resource*
_output_shapes
:F*
dtype0º
%batch_normalization_627/batchnorm/subSub:batch_normalization_627/batchnorm/ReadVariableOp_2:value:0+batch_normalization_627/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Fº
'batch_normalization_627/batchnorm/add_1AddV2+batch_normalization_627/batchnorm/mul_1:z:0)batch_normalization_627/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
leaky_re_lu_627/LeakyRelu	LeakyRelu+batch_normalization_627/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
alpha%>
dense_698/MatMul/ReadVariableOpReadVariableOp(dense_698_matmul_readvariableop_resource*
_output_shapes

:F*
dtype0
dense_698/MatMulMatMul'leaky_re_lu_627/LeakyRelu:activations:0'dense_698/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_698/BiasAdd/ReadVariableOpReadVariableOp)dense_698_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_698/BiasAddBiasAdddense_698/MatMul:product:0(dense_698/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/dense_695/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_695_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
 dense_695/kernel/Regularizer/AbsAbs7dense_695/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:s
"dense_695/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_695/kernel/Regularizer/SumSum$dense_695/kernel/Regularizer/Abs:y:0+dense_695/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_695/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_695/kernel/Regularizer/mulMul+dense_695/kernel/Regularizer/mul/x:output:0)dense_695/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/dense_696/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_696_matmul_readvariableop_resource*
_output_shapes

:7*
dtype0
 dense_696/kernel/Regularizer/AbsAbs7dense_696/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7s
"dense_696/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_696/kernel/Regularizer/SumSum$dense_696/kernel/Regularizer/Abs:y:0+dense_696/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_696/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_696/kernel/Regularizer/mulMul+dense_696/kernel/Regularizer/mul/x:output:0)dense_696/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/dense_697/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_697_matmul_readvariableop_resource*
_output_shapes

:7F*
dtype0
 dense_697/kernel/Regularizer/AbsAbs7dense_697/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7Fs
"dense_697/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_697/kernel/Regularizer/SumSum$dense_697/kernel/Regularizer/Abs:y:0+dense_697/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_697/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 dense_697/kernel/Regularizer/mulMul+dense_697/kernel/Regularizer/mul/x:output:0)dense_697/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_698/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
NoOpNoOp1^batch_normalization_625/batchnorm/ReadVariableOp3^batch_normalization_625/batchnorm/ReadVariableOp_13^batch_normalization_625/batchnorm/ReadVariableOp_25^batch_normalization_625/batchnorm/mul/ReadVariableOp1^batch_normalization_626/batchnorm/ReadVariableOp3^batch_normalization_626/batchnorm/ReadVariableOp_13^batch_normalization_626/batchnorm/ReadVariableOp_25^batch_normalization_626/batchnorm/mul/ReadVariableOp1^batch_normalization_627/batchnorm/ReadVariableOp3^batch_normalization_627/batchnorm/ReadVariableOp_13^batch_normalization_627/batchnorm/ReadVariableOp_25^batch_normalization_627/batchnorm/mul/ReadVariableOp!^dense_695/BiasAdd/ReadVariableOp ^dense_695/MatMul/ReadVariableOp0^dense_695/kernel/Regularizer/Abs/ReadVariableOp!^dense_696/BiasAdd/ReadVariableOp ^dense_696/MatMul/ReadVariableOp0^dense_696/kernel/Regularizer/Abs/ReadVariableOp!^dense_697/BiasAdd/ReadVariableOp ^dense_697/MatMul/ReadVariableOp0^dense_697/kernel/Regularizer/Abs/ReadVariableOp!^dense_698/BiasAdd/ReadVariableOp ^dense_698/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : 2d
0batch_normalization_625/batchnorm/ReadVariableOp0batch_normalization_625/batchnorm/ReadVariableOp2h
2batch_normalization_625/batchnorm/ReadVariableOp_12batch_normalization_625/batchnorm/ReadVariableOp_12h
2batch_normalization_625/batchnorm/ReadVariableOp_22batch_normalization_625/batchnorm/ReadVariableOp_22l
4batch_normalization_625/batchnorm/mul/ReadVariableOp4batch_normalization_625/batchnorm/mul/ReadVariableOp2d
0batch_normalization_626/batchnorm/ReadVariableOp0batch_normalization_626/batchnorm/ReadVariableOp2h
2batch_normalization_626/batchnorm/ReadVariableOp_12batch_normalization_626/batchnorm/ReadVariableOp_12h
2batch_normalization_626/batchnorm/ReadVariableOp_22batch_normalization_626/batchnorm/ReadVariableOp_22l
4batch_normalization_626/batchnorm/mul/ReadVariableOp4batch_normalization_626/batchnorm/mul/ReadVariableOp2d
0batch_normalization_627/batchnorm/ReadVariableOp0batch_normalization_627/batchnorm/ReadVariableOp2h
2batch_normalization_627/batchnorm/ReadVariableOp_12batch_normalization_627/batchnorm/ReadVariableOp_12h
2batch_normalization_627/batchnorm/ReadVariableOp_22batch_normalization_627/batchnorm/ReadVariableOp_22l
4batch_normalization_627/batchnorm/mul/ReadVariableOp4batch_normalization_627/batchnorm/mul/ReadVariableOp2D
 dense_695/BiasAdd/ReadVariableOp dense_695/BiasAdd/ReadVariableOp2B
dense_695/MatMul/ReadVariableOpdense_695/MatMul/ReadVariableOp2b
/dense_695/kernel/Regularizer/Abs/ReadVariableOp/dense_695/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_696/BiasAdd/ReadVariableOp dense_696/BiasAdd/ReadVariableOp2B
dense_696/MatMul/ReadVariableOpdense_696/MatMul/ReadVariableOp2b
/dense_696/kernel/Regularizer/Abs/ReadVariableOp/dense_696/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_697/BiasAdd/ReadVariableOp dense_697/BiasAdd/ReadVariableOp2B
dense_697/MatMul/ReadVariableOpdense_697/MatMul/ReadVariableOp2b
/dense_697/kernel/Regularizer/Abs/ReadVariableOp/dense_697/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_698/BiasAdd/ReadVariableOp dense_698/BiasAdd/ReadVariableOp2B
dense_698/MatMul/ReadVariableOpdense_698/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ð

/__inference_sequential_70_layer_call_fn_1237261

inputs
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:7
	unknown_8:7
	unknown_9:7

unknown_10:7

unknown_11:7

unknown_12:7

unknown_13:7F

unknown_14:F

unknown_15:F

unknown_16:F

unknown_17:F

unknown_18:F

unknown_19:F

unknown_20:
identity¢StatefulPartitionedCallæ
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_70_layer_call_and_return_conditional_losses_1236887o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
%
í
T__inference_batch_normalization_627_layer_call_and_return_conditional_losses_1236478

inputs5
'assignmovingavg_readvariableop_resource:F7
)assignmovingavg_1_readvariableop_resource:F3
%batchnorm_mul_readvariableop_resource:F/
!batchnorm_readvariableop_resource:F
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:F*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:F
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:F*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:F*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:F*
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
:F*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Fx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:F¬
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
:F*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:F~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:F´
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
:FP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:F~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Fc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Fv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:F*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Fr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
øS
ª
J__inference_sequential_70_layer_call_and_return_conditional_losses_1236652

inputs
normalization_70_sub_y
normalization_70_sqrt_x#
dense_695_1236520:
dense_695_1236522:-
batch_normalization_625_1236525:-
batch_normalization_625_1236527:-
batch_normalization_625_1236529:-
batch_normalization_625_1236531:#
dense_696_1236558:7
dense_696_1236560:7-
batch_normalization_626_1236563:7-
batch_normalization_626_1236565:7-
batch_normalization_626_1236567:7-
batch_normalization_626_1236569:7#
dense_697_1236596:7F
dense_697_1236598:F-
batch_normalization_627_1236601:F-
batch_normalization_627_1236603:F-
batch_normalization_627_1236605:F-
batch_normalization_627_1236607:F#
dense_698_1236628:F
dense_698_1236630:
identity¢/batch_normalization_625/StatefulPartitionedCall¢/batch_normalization_626/StatefulPartitionedCall¢/batch_normalization_627/StatefulPartitionedCall¢!dense_695/StatefulPartitionedCall¢/dense_695/kernel/Regularizer/Abs/ReadVariableOp¢!dense_696/StatefulPartitionedCall¢/dense_696/kernel/Regularizer/Abs/ReadVariableOp¢!dense_697/StatefulPartitionedCall¢/dense_697/kernel/Regularizer/Abs/ReadVariableOp¢!dense_698/StatefulPartitionedCallm
normalization_70/subSubinputsnormalization_70_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_70/SqrtSqrtnormalization_70_sqrt_x*
T0*
_output_shapes

:_
normalization_70/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_70/MaximumMaximumnormalization_70/Sqrt:y:0#normalization_70/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_70/truedivRealDivnormalization_70/sub:z:0normalization_70/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_695/StatefulPartitionedCallStatefulPartitionedCallnormalization_70/truediv:z:0dense_695_1236520dense_695_1236522*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_695_layer_call_and_return_conditional_losses_1236519
/batch_normalization_625/StatefulPartitionedCallStatefulPartitionedCall*dense_695/StatefulPartitionedCall:output:0batch_normalization_625_1236525batch_normalization_625_1236527batch_normalization_625_1236529batch_normalization_625_1236531*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_625_layer_call_and_return_conditional_losses_1236267ù
leaky_re_lu_625/PartitionedCallPartitionedCall8batch_normalization_625/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_625_layer_call_and_return_conditional_losses_1236539
!dense_696/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_625/PartitionedCall:output:0dense_696_1236558dense_696_1236560*
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
F__inference_dense_696_layer_call_and_return_conditional_losses_1236557
/batch_normalization_626/StatefulPartitionedCallStatefulPartitionedCall*dense_696/StatefulPartitionedCall:output:0batch_normalization_626_1236563batch_normalization_626_1236565batch_normalization_626_1236567batch_normalization_626_1236569*
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
T__inference_batch_normalization_626_layer_call_and_return_conditional_losses_1236349ù
leaky_re_lu_626/PartitionedCallPartitionedCall8batch_normalization_626/StatefulPartitionedCall:output:0*
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
L__inference_leaky_re_lu_626_layer_call_and_return_conditional_losses_1236577
!dense_697/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_626/PartitionedCall:output:0dense_697_1236596dense_697_1236598*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_697_layer_call_and_return_conditional_losses_1236595
/batch_normalization_627/StatefulPartitionedCallStatefulPartitionedCall*dense_697/StatefulPartitionedCall:output:0batch_normalization_627_1236601batch_normalization_627_1236603batch_normalization_627_1236605batch_normalization_627_1236607*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_627_layer_call_and_return_conditional_losses_1236431ù
leaky_re_lu_627/PartitionedCallPartitionedCall8batch_normalization_627/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_627_layer_call_and_return_conditional_losses_1236615
!dense_698/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_627/PartitionedCall:output:0dense_698_1236628dense_698_1236630*
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
F__inference_dense_698_layer_call_and_return_conditional_losses_1236627
/dense_695/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_695_1236520*
_output_shapes

:*
dtype0
 dense_695/kernel/Regularizer/AbsAbs7dense_695/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:s
"dense_695/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_695/kernel/Regularizer/SumSum$dense_695/kernel/Regularizer/Abs:y:0+dense_695/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_695/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_695/kernel/Regularizer/mulMul+dense_695/kernel/Regularizer/mul/x:output:0)dense_695/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/dense_696/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_696_1236558*
_output_shapes

:7*
dtype0
 dense_696/kernel/Regularizer/AbsAbs7dense_696/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7s
"dense_696/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_696/kernel/Regularizer/SumSum$dense_696/kernel/Regularizer/Abs:y:0+dense_696/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_696/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_696/kernel/Regularizer/mulMul+dense_696/kernel/Regularizer/mul/x:output:0)dense_696/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/dense_697/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_697_1236596*
_output_shapes

:7F*
dtype0
 dense_697/kernel/Regularizer/AbsAbs7dense_697/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7Fs
"dense_697/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_697/kernel/Regularizer/SumSum$dense_697/kernel/Regularizer/Abs:y:0+dense_697/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_697/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 dense_697/kernel/Regularizer/mulMul+dense_697/kernel/Regularizer/mul/x:output:0)dense_697/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_698/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_625/StatefulPartitionedCall0^batch_normalization_626/StatefulPartitionedCall0^batch_normalization_627/StatefulPartitionedCall"^dense_695/StatefulPartitionedCall0^dense_695/kernel/Regularizer/Abs/ReadVariableOp"^dense_696/StatefulPartitionedCall0^dense_696/kernel/Regularizer/Abs/ReadVariableOp"^dense_697/StatefulPartitionedCall0^dense_697/kernel/Regularizer/Abs/ReadVariableOp"^dense_698/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_625/StatefulPartitionedCall/batch_normalization_625/StatefulPartitionedCall2b
/batch_normalization_626/StatefulPartitionedCall/batch_normalization_626/StatefulPartitionedCall2b
/batch_normalization_627/StatefulPartitionedCall/batch_normalization_627/StatefulPartitionedCall2F
!dense_695/StatefulPartitionedCall!dense_695/StatefulPartitionedCall2b
/dense_695/kernel/Regularizer/Abs/ReadVariableOp/dense_695/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_696/StatefulPartitionedCall!dense_696/StatefulPartitionedCall2b
/dense_696/kernel/Regularizer/Abs/ReadVariableOp/dense_696/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_697/StatefulPartitionedCall!dense_697/StatefulPartitionedCall2b
/dense_697/kernel/Regularizer/Abs/ReadVariableOp/dense_697/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_698/StatefulPartitionedCall!dense_698/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
%
í
T__inference_batch_normalization_626_layer_call_and_return_conditional_losses_1236396

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
Î
©
F__inference_dense_695_layer_call_and_return_conditional_losses_1236519

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_695/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/dense_695/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
 dense_695/kernel/Regularizer/AbsAbs7dense_695/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:s
"dense_695/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_695/kernel/Regularizer/SumSum$dense_695/kernel/Regularizer/Abs:y:0+dense_695/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_695/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_695/kernel/Regularizer/mulMul+dense_695/kernel/Regularizer/mul/x:output:0)dense_695/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_695/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_695/kernel/Regularizer/Abs/ReadVariableOp/dense_695/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï'
Ó
__inference_adapt_step_1237609
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
%
í
T__inference_batch_normalization_626_layer_call_and_return_conditional_losses_1237841

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
®
Ô
9__inference_batch_normalization_626_layer_call_fn_1237774

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
T__inference_batch_normalization_626_layer_call_and_return_conditional_losses_1236349o
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
Î
©
F__inference_dense_697_layer_call_and_return_conditional_losses_1236595

inputs0
matmul_readvariableop_resource:7F-
biasadd_readvariableop_resource:F
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_697/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7F*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
/dense_697/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7F*
dtype0
 dense_697/kernel/Regularizer/AbsAbs7dense_697/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7Fs
"dense_697/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_697/kernel/Regularizer/SumSum$dense_697/kernel/Regularizer/Abs:y:0+dense_697/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_697/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 dense_697/kernel/Regularizer/mulMul+dense_697/kernel/Regularizer/mul/x:output:0)dense_697/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF©
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_697/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_697/kernel/Regularizer/Abs/ReadVariableOp/dense_697/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
Î
©
F__inference_dense_696_layer_call_and_return_conditional_losses_1236557

inputs0
matmul_readvariableop_resource:7-
biasadd_readvariableop_resource:7
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_696/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7*
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
:ÿÿÿÿÿÿÿÿÿ7
/dense_696/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7*
dtype0
 dense_696/kernel/Regularizer/AbsAbs7dense_696/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7s
"dense_696/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_696/kernel/Regularizer/SumSum$dense_696/kernel/Regularizer/Abs:y:0+dense_696/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_696/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_696/kernel/Regularizer/mulMul+dense_696/kernel/Regularizer/mul/x:output:0)dense_696/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7©
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_696/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_696/kernel/Regularizer/Abs/ReadVariableOp/dense_696/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_625_layer_call_and_return_conditional_losses_1236539

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

£
/__inference_sequential_70_layer_call_fn_1236983
normalization_70_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:7
	unknown_8:7
	unknown_9:7

unknown_10:7

unknown_11:7

unknown_12:7

unknown_13:7F

unknown_14:F

unknown_15:F

unknown_16:F

unknown_17:F

unknown_18:F

unknown_19:F

unknown_20:
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallnormalization_70_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_70_layer_call_and_return_conditional_losses_1236887o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_70_input:$ 

_output_shapes

::$ 

_output_shapes

:
æ
h
L__inference_leaky_re_lu_625_layer_call_and_return_conditional_losses_1237730

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
%
#__inference__traced_restore_1238401
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 5
#assignvariableop_3_dense_695_kernel:/
!assignvariableop_4_dense_695_bias:>
0assignvariableop_5_batch_normalization_625_gamma:=
/assignvariableop_6_batch_normalization_625_beta:D
6assignvariableop_7_batch_normalization_625_moving_mean:H
:assignvariableop_8_batch_normalization_625_moving_variance:5
#assignvariableop_9_dense_696_kernel:70
"assignvariableop_10_dense_696_bias:7?
1assignvariableop_11_batch_normalization_626_gamma:7>
0assignvariableop_12_batch_normalization_626_beta:7E
7assignvariableop_13_batch_normalization_626_moving_mean:7I
;assignvariableop_14_batch_normalization_626_moving_variance:76
$assignvariableop_15_dense_697_kernel:7F0
"assignvariableop_16_dense_697_bias:F?
1assignvariableop_17_batch_normalization_627_gamma:F>
0assignvariableop_18_batch_normalization_627_beta:FE
7assignvariableop_19_batch_normalization_627_moving_mean:FI
;assignvariableop_20_batch_normalization_627_moving_variance:F6
$assignvariableop_21_dense_698_kernel:F0
"assignvariableop_22_dense_698_bias:'
assignvariableop_23_adam_iter:	 )
assignvariableop_24_adam_beta_1: )
assignvariableop_25_adam_beta_2: (
assignvariableop_26_adam_decay: #
assignvariableop_27_total: %
assignvariableop_28_count_1: =
+assignvariableop_29_adam_dense_695_kernel_m:7
)assignvariableop_30_adam_dense_695_bias_m:F
8assignvariableop_31_adam_batch_normalization_625_gamma_m:E
7assignvariableop_32_adam_batch_normalization_625_beta_m:=
+assignvariableop_33_adam_dense_696_kernel_m:77
)assignvariableop_34_adam_dense_696_bias_m:7F
8assignvariableop_35_adam_batch_normalization_626_gamma_m:7E
7assignvariableop_36_adam_batch_normalization_626_beta_m:7=
+assignvariableop_37_adam_dense_697_kernel_m:7F7
)assignvariableop_38_adam_dense_697_bias_m:FF
8assignvariableop_39_adam_batch_normalization_627_gamma_m:FE
7assignvariableop_40_adam_batch_normalization_627_beta_m:F=
+assignvariableop_41_adam_dense_698_kernel_m:F7
)assignvariableop_42_adam_dense_698_bias_m:=
+assignvariableop_43_adam_dense_695_kernel_v:7
)assignvariableop_44_adam_dense_695_bias_v:F
8assignvariableop_45_adam_batch_normalization_625_gamma_v:E
7assignvariableop_46_adam_batch_normalization_625_beta_v:=
+assignvariableop_47_adam_dense_696_kernel_v:77
)assignvariableop_48_adam_dense_696_bias_v:7F
8assignvariableop_49_adam_batch_normalization_626_gamma_v:7E
7assignvariableop_50_adam_batch_normalization_626_beta_v:7=
+assignvariableop_51_adam_dense_697_kernel_v:7F7
)assignvariableop_52_adam_dense_697_bias_v:FF
8assignvariableop_53_adam_batch_normalization_627_gamma_v:FE
7assignvariableop_54_adam_batch_normalization_627_beta_v:F=
+assignvariableop_55_adam_dense_698_kernel_v:F7
)assignvariableop_56_adam_dense_698_bias_v:
identity_58¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ø
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*þ
valueôBñ:B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHå
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ã
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*þ
_output_shapesë
è::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:		[
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
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_695_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_695_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp0assignvariableop_5_batch_normalization_625_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_625_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_625_moving_meanIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_8AssignVariableOp:assignvariableop_8_batch_normalization_625_moving_varianceIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_696_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_696_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_11AssignVariableOp1assignvariableop_11_batch_normalization_626_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_626_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_626_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_14AssignVariableOp;assignvariableop_14_batch_normalization_626_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_697_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_697_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_17AssignVariableOp1assignvariableop_17_batch_normalization_627_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_627_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_19AssignVariableOp7assignvariableop_19_batch_normalization_627_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_20AssignVariableOp;assignvariableop_20_batch_normalization_627_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp$assignvariableop_21_dense_698_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_698_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_iterIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_beta_2Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_decayIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_695_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_695_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_31AssignVariableOp8assignvariableop_31_adam_batch_normalization_625_gamma_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_32AssignVariableOp7assignvariableop_32_adam_batch_normalization_625_beta_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_696_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_696_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_35AssignVariableOp8assignvariableop_35_adam_batch_normalization_626_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_36AssignVariableOp7assignvariableop_36_adam_batch_normalization_626_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_dense_697_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_697_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_39AssignVariableOp8assignvariableop_39_adam_batch_normalization_627_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_40AssignVariableOp7assignvariableop_40_adam_batch_normalization_627_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_dense_698_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_dense_698_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_dense_695_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_dense_695_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_45AssignVariableOp8assignvariableop_45_adam_batch_normalization_625_gamma_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_46AssignVariableOp7assignvariableop_46_adam_batch_normalization_625_beta_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_dense_696_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_696_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_49AssignVariableOp8assignvariableop_49_adam_batch_normalization_626_gamma_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_50AssignVariableOp7assignvariableop_50_adam_batch_normalization_626_beta_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_dense_697_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_dense_697_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_53AssignVariableOp8assignvariableop_53_adam_batch_normalization_627_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_54AssignVariableOp7assignvariableop_54_adam_batch_normalization_627_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_dense_698_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_dense_698_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 µ

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: ¢

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ñ
³
T__inference_batch_normalization_625_layer_call_and_return_conditional_losses_1237686

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
Þ
J__inference_sequential_70_layer_call_and_return_conditional_losses_1237511

inputs
normalization_70_sub_y
normalization_70_sqrt_x:
(dense_695_matmul_readvariableop_resource:7
)dense_695_biasadd_readvariableop_resource:M
?batch_normalization_625_assignmovingavg_readvariableop_resource:O
Abatch_normalization_625_assignmovingavg_1_readvariableop_resource:K
=batch_normalization_625_batchnorm_mul_readvariableop_resource:G
9batch_normalization_625_batchnorm_readvariableop_resource::
(dense_696_matmul_readvariableop_resource:77
)dense_696_biasadd_readvariableop_resource:7M
?batch_normalization_626_assignmovingavg_readvariableop_resource:7O
Abatch_normalization_626_assignmovingavg_1_readvariableop_resource:7K
=batch_normalization_626_batchnorm_mul_readvariableop_resource:7G
9batch_normalization_626_batchnorm_readvariableop_resource:7:
(dense_697_matmul_readvariableop_resource:7F7
)dense_697_biasadd_readvariableop_resource:FM
?batch_normalization_627_assignmovingavg_readvariableop_resource:FO
Abatch_normalization_627_assignmovingavg_1_readvariableop_resource:FK
=batch_normalization_627_batchnorm_mul_readvariableop_resource:FG
9batch_normalization_627_batchnorm_readvariableop_resource:F:
(dense_698_matmul_readvariableop_resource:F7
)dense_698_biasadd_readvariableop_resource:
identity¢'batch_normalization_625/AssignMovingAvg¢6batch_normalization_625/AssignMovingAvg/ReadVariableOp¢)batch_normalization_625/AssignMovingAvg_1¢8batch_normalization_625/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_625/batchnorm/ReadVariableOp¢4batch_normalization_625/batchnorm/mul/ReadVariableOp¢'batch_normalization_626/AssignMovingAvg¢6batch_normalization_626/AssignMovingAvg/ReadVariableOp¢)batch_normalization_626/AssignMovingAvg_1¢8batch_normalization_626/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_626/batchnorm/ReadVariableOp¢4batch_normalization_626/batchnorm/mul/ReadVariableOp¢'batch_normalization_627/AssignMovingAvg¢6batch_normalization_627/AssignMovingAvg/ReadVariableOp¢)batch_normalization_627/AssignMovingAvg_1¢8batch_normalization_627/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_627/batchnorm/ReadVariableOp¢4batch_normalization_627/batchnorm/mul/ReadVariableOp¢ dense_695/BiasAdd/ReadVariableOp¢dense_695/MatMul/ReadVariableOp¢/dense_695/kernel/Regularizer/Abs/ReadVariableOp¢ dense_696/BiasAdd/ReadVariableOp¢dense_696/MatMul/ReadVariableOp¢/dense_696/kernel/Regularizer/Abs/ReadVariableOp¢ dense_697/BiasAdd/ReadVariableOp¢dense_697/MatMul/ReadVariableOp¢/dense_697/kernel/Regularizer/Abs/ReadVariableOp¢ dense_698/BiasAdd/ReadVariableOp¢dense_698/MatMul/ReadVariableOpm
normalization_70/subSubinputsnormalization_70_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_70/SqrtSqrtnormalization_70_sqrt_x*
T0*
_output_shapes

:_
normalization_70/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_70/MaximumMaximumnormalization_70/Sqrt:y:0#normalization_70/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_70/truedivRealDivnormalization_70/sub:z:0normalization_70/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_695/MatMul/ReadVariableOpReadVariableOp(dense_695_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_695/MatMulMatMulnormalization_70/truediv:z:0'dense_695/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_695/BiasAdd/ReadVariableOpReadVariableOp)dense_695_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_695/BiasAddBiasAdddense_695/MatMul:product:0(dense_695/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization_625/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_625/moments/meanMeandense_695/BiasAdd:output:0?batch_normalization_625/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
,batch_normalization_625/moments/StopGradientStopGradient-batch_normalization_625/moments/mean:output:0*
T0*
_output_shapes

:Ë
1batch_normalization_625/moments/SquaredDifferenceSquaredDifferencedense_695/BiasAdd:output:05batch_normalization_625/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:batch_normalization_625/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_625/moments/varianceMean5batch_normalization_625/moments/SquaredDifference:z:0Cbatch_normalization_625/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
'batch_normalization_625/moments/SqueezeSqueeze-batch_normalization_625/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 £
)batch_normalization_625/moments/Squeeze_1Squeeze1batch_normalization_625/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-batch_normalization_625/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_625/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_625_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0É
+batch_normalization_625/AssignMovingAvg/subSub>batch_normalization_625/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_625/moments/Squeeze:output:0*
T0*
_output_shapes
:À
+batch_normalization_625/AssignMovingAvg/mulMul/batch_normalization_625/AssignMovingAvg/sub:z:06batch_normalization_625/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:
'batch_normalization_625/AssignMovingAvgAssignSubVariableOp?batch_normalization_625_assignmovingavg_readvariableop_resource/batch_normalization_625/AssignMovingAvg/mul:z:07^batch_normalization_625/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_625/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_625/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_625_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ï
-batch_normalization_625/AssignMovingAvg_1/subSub@batch_normalization_625/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_625/moments/Squeeze_1:output:0*
T0*
_output_shapes
:Æ
-batch_normalization_625/AssignMovingAvg_1/mulMul1batch_normalization_625/AssignMovingAvg_1/sub:z:08batch_normalization_625/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
)batch_normalization_625/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_625_assignmovingavg_1_readvariableop_resource1batch_normalization_625/AssignMovingAvg_1/mul:z:09^batch_normalization_625/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_625/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_625/batchnorm/addAddV22batch_normalization_625/moments/Squeeze_1:output:00batch_normalization_625/batchnorm/add/y:output:0*
T0*
_output_shapes
:
'batch_normalization_625/batchnorm/RsqrtRsqrt)batch_normalization_625/batchnorm/add:z:0*
T0*
_output_shapes
:®
4batch_normalization_625/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_625_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0¼
%batch_normalization_625/batchnorm/mulMul+batch_normalization_625/batchnorm/Rsqrt:y:0<batch_normalization_625/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:§
'batch_normalization_625/batchnorm/mul_1Muldense_695/BiasAdd:output:0)batch_normalization_625/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
'batch_normalization_625/batchnorm/mul_2Mul0batch_normalization_625/moments/Squeeze:output:0)batch_normalization_625/batchnorm/mul:z:0*
T0*
_output_shapes
:¦
0batch_normalization_625/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_625_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¸
%batch_normalization_625/batchnorm/subSub8batch_normalization_625/batchnorm/ReadVariableOp:value:0+batch_normalization_625/batchnorm/mul_2:z:0*
T0*
_output_shapes
:º
'batch_normalization_625/batchnorm/add_1AddV2+batch_normalization_625/batchnorm/mul_1:z:0)batch_normalization_625/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
leaky_re_lu_625/LeakyRelu	LeakyRelu+batch_normalization_625/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>
dense_696/MatMul/ReadVariableOpReadVariableOp(dense_696_matmul_readvariableop_resource*
_output_shapes

:7*
dtype0
dense_696/MatMulMatMul'leaky_re_lu_625/LeakyRelu:activations:0'dense_696/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 dense_696/BiasAdd/ReadVariableOpReadVariableOp)dense_696_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0
dense_696/BiasAddBiasAdddense_696/MatMul:product:0(dense_696/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
6batch_normalization_626/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_626/moments/meanMeandense_696/BiasAdd:output:0?batch_normalization_626/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(
,batch_normalization_626/moments/StopGradientStopGradient-batch_normalization_626/moments/mean:output:0*
T0*
_output_shapes

:7Ë
1batch_normalization_626/moments/SquaredDifferenceSquaredDifferencedense_696/BiasAdd:output:05batch_normalization_626/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
:batch_normalization_626/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_626/moments/varianceMean5batch_normalization_626/moments/SquaredDifference:z:0Cbatch_normalization_626/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:7*
	keep_dims(
'batch_normalization_626/moments/SqueezeSqueeze-batch_normalization_626/moments/mean:output:0*
T0*
_output_shapes
:7*
squeeze_dims
 £
)batch_normalization_626/moments/Squeeze_1Squeeze1batch_normalization_626/moments/variance:output:0*
T0*
_output_shapes
:7*
squeeze_dims
 r
-batch_normalization_626/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_626/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_626_assignmovingavg_readvariableop_resource*
_output_shapes
:7*
dtype0É
+batch_normalization_626/AssignMovingAvg/subSub>batch_normalization_626/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_626/moments/Squeeze:output:0*
T0*
_output_shapes
:7À
+batch_normalization_626/AssignMovingAvg/mulMul/batch_normalization_626/AssignMovingAvg/sub:z:06batch_normalization_626/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:7
'batch_normalization_626/AssignMovingAvgAssignSubVariableOp?batch_normalization_626_assignmovingavg_readvariableop_resource/batch_normalization_626/AssignMovingAvg/mul:z:07^batch_normalization_626/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_626/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_626/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_626_assignmovingavg_1_readvariableop_resource*
_output_shapes
:7*
dtype0Ï
-batch_normalization_626/AssignMovingAvg_1/subSub@batch_normalization_626/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_626/moments/Squeeze_1:output:0*
T0*
_output_shapes
:7Æ
-batch_normalization_626/AssignMovingAvg_1/mulMul1batch_normalization_626/AssignMovingAvg_1/sub:z:08batch_normalization_626/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:7
)batch_normalization_626/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_626_assignmovingavg_1_readvariableop_resource1batch_normalization_626/AssignMovingAvg_1/mul:z:09^batch_normalization_626/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_626/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_626/batchnorm/addAddV22batch_normalization_626/moments/Squeeze_1:output:00batch_normalization_626/batchnorm/add/y:output:0*
T0*
_output_shapes
:7
'batch_normalization_626/batchnorm/RsqrtRsqrt)batch_normalization_626/batchnorm/add:z:0*
T0*
_output_shapes
:7®
4batch_normalization_626/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_626_batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0¼
%batch_normalization_626/batchnorm/mulMul+batch_normalization_626/batchnorm/Rsqrt:y:0<batch_normalization_626/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7§
'batch_normalization_626/batchnorm/mul_1Muldense_696/BiasAdd:output:0)batch_normalization_626/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7°
'batch_normalization_626/batchnorm/mul_2Mul0batch_normalization_626/moments/Squeeze:output:0)batch_normalization_626/batchnorm/mul:z:0*
T0*
_output_shapes
:7¦
0batch_normalization_626/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_626_batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0¸
%batch_normalization_626/batchnorm/subSub8batch_normalization_626/batchnorm/ReadVariableOp:value:0+batch_normalization_626/batchnorm/mul_2:z:0*
T0*
_output_shapes
:7º
'batch_normalization_626/batchnorm/add_1AddV2+batch_normalization_626/batchnorm/mul_1:z:0)batch_normalization_626/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
leaky_re_lu_626/LeakyRelu	LeakyRelu+batch_normalization_626/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>
dense_697/MatMul/ReadVariableOpReadVariableOp(dense_697_matmul_readvariableop_resource*
_output_shapes

:7F*
dtype0
dense_697/MatMulMatMul'leaky_re_lu_626/LeakyRelu:activations:0'dense_697/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 dense_697/BiasAdd/ReadVariableOpReadVariableOp)dense_697_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype0
dense_697/BiasAddBiasAdddense_697/MatMul:product:0(dense_697/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
6batch_normalization_627/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
$batch_normalization_627/moments/meanMeandense_697/BiasAdd:output:0?batch_normalization_627/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:F*
	keep_dims(
,batch_normalization_627/moments/StopGradientStopGradient-batch_normalization_627/moments/mean:output:0*
T0*
_output_shapes

:FË
1batch_normalization_627/moments/SquaredDifferenceSquaredDifferencedense_697/BiasAdd:output:05batch_normalization_627/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
:batch_normalization_627/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: æ
(batch_normalization_627/moments/varianceMean5batch_normalization_627/moments/SquaredDifference:z:0Cbatch_normalization_627/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:F*
	keep_dims(
'batch_normalization_627/moments/SqueezeSqueeze-batch_normalization_627/moments/mean:output:0*
T0*
_output_shapes
:F*
squeeze_dims
 £
)batch_normalization_627/moments/Squeeze_1Squeeze1batch_normalization_627/moments/variance:output:0*
T0*
_output_shapes
:F*
squeeze_dims
 r
-batch_normalization_627/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_627/AssignMovingAvg/ReadVariableOpReadVariableOp?batch_normalization_627_assignmovingavg_readvariableop_resource*
_output_shapes
:F*
dtype0É
+batch_normalization_627/AssignMovingAvg/subSub>batch_normalization_627/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_627/moments/Squeeze:output:0*
T0*
_output_shapes
:FÀ
+batch_normalization_627/AssignMovingAvg/mulMul/batch_normalization_627/AssignMovingAvg/sub:z:06batch_normalization_627/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:F
'batch_normalization_627/AssignMovingAvgAssignSubVariableOp?batch_normalization_627_assignmovingavg_readvariableop_resource/batch_normalization_627/AssignMovingAvg/mul:z:07^batch_normalization_627/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/batch_normalization_627/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¶
8batch_normalization_627/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatch_normalization_627_assignmovingavg_1_readvariableop_resource*
_output_shapes
:F*
dtype0Ï
-batch_normalization_627/AssignMovingAvg_1/subSub@batch_normalization_627/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_627/moments/Squeeze_1:output:0*
T0*
_output_shapes
:FÆ
-batch_normalization_627/AssignMovingAvg_1/mulMul1batch_normalization_627/AssignMovingAvg_1/sub:z:08batch_normalization_627/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:F
)batch_normalization_627/AssignMovingAvg_1AssignSubVariableOpAbatch_normalization_627_assignmovingavg_1_readvariableop_resource1batch_normalization_627/AssignMovingAvg_1/mul:z:09^batch_normalization_627/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'batch_normalization_627/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
%batch_normalization_627/batchnorm/addAddV22batch_normalization_627/moments/Squeeze_1:output:00batch_normalization_627/batchnorm/add/y:output:0*
T0*
_output_shapes
:F
'batch_normalization_627/batchnorm/RsqrtRsqrt)batch_normalization_627/batchnorm/add:z:0*
T0*
_output_shapes
:F®
4batch_normalization_627/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_627_batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype0¼
%batch_normalization_627/batchnorm/mulMul+batch_normalization_627/batchnorm/Rsqrt:y:0<batch_normalization_627/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:F§
'batch_normalization_627/batchnorm/mul_1Muldense_697/BiasAdd:output:0)batch_normalization_627/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF°
'batch_normalization_627/batchnorm/mul_2Mul0batch_normalization_627/moments/Squeeze:output:0)batch_normalization_627/batchnorm/mul:z:0*
T0*
_output_shapes
:F¦
0batch_normalization_627/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_627_batchnorm_readvariableop_resource*
_output_shapes
:F*
dtype0¸
%batch_normalization_627/batchnorm/subSub8batch_normalization_627/batchnorm/ReadVariableOp:value:0+batch_normalization_627/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Fº
'batch_normalization_627/batchnorm/add_1AddV2+batch_normalization_627/batchnorm/mul_1:z:0)batch_normalization_627/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
leaky_re_lu_627/LeakyRelu	LeakyRelu+batch_normalization_627/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
alpha%>
dense_698/MatMul/ReadVariableOpReadVariableOp(dense_698_matmul_readvariableop_resource*
_output_shapes

:F*
dtype0
dense_698/MatMulMatMul'leaky_re_lu_627/LeakyRelu:activations:0'dense_698/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_698/BiasAdd/ReadVariableOpReadVariableOp)dense_698_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_698/BiasAddBiasAdddense_698/MatMul:product:0(dense_698/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/dense_695/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_695_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
 dense_695/kernel/Regularizer/AbsAbs7dense_695/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:s
"dense_695/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_695/kernel/Regularizer/SumSum$dense_695/kernel/Regularizer/Abs:y:0+dense_695/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_695/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_695/kernel/Regularizer/mulMul+dense_695/kernel/Regularizer/mul/x:output:0)dense_695/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/dense_696/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_696_matmul_readvariableop_resource*
_output_shapes

:7*
dtype0
 dense_696/kernel/Regularizer/AbsAbs7dense_696/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7s
"dense_696/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_696/kernel/Regularizer/SumSum$dense_696/kernel/Regularizer/Abs:y:0+dense_696/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_696/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_696/kernel/Regularizer/mulMul+dense_696/kernel/Regularizer/mul/x:output:0)dense_696/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/dense_697/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_697_matmul_readvariableop_resource*
_output_shapes

:7F*
dtype0
 dense_697/kernel/Regularizer/AbsAbs7dense_697/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7Fs
"dense_697/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_697/kernel/Regularizer/SumSum$dense_697/kernel/Regularizer/Abs:y:0+dense_697/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_697/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 dense_697/kernel/Regularizer/mulMul+dense_697/kernel/Regularizer/mul/x:output:0)dense_697/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_698/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp(^batch_normalization_625/AssignMovingAvg7^batch_normalization_625/AssignMovingAvg/ReadVariableOp*^batch_normalization_625/AssignMovingAvg_19^batch_normalization_625/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_625/batchnorm/ReadVariableOp5^batch_normalization_625/batchnorm/mul/ReadVariableOp(^batch_normalization_626/AssignMovingAvg7^batch_normalization_626/AssignMovingAvg/ReadVariableOp*^batch_normalization_626/AssignMovingAvg_19^batch_normalization_626/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_626/batchnorm/ReadVariableOp5^batch_normalization_626/batchnorm/mul/ReadVariableOp(^batch_normalization_627/AssignMovingAvg7^batch_normalization_627/AssignMovingAvg/ReadVariableOp*^batch_normalization_627/AssignMovingAvg_19^batch_normalization_627/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_627/batchnorm/ReadVariableOp5^batch_normalization_627/batchnorm/mul/ReadVariableOp!^dense_695/BiasAdd/ReadVariableOp ^dense_695/MatMul/ReadVariableOp0^dense_695/kernel/Regularizer/Abs/ReadVariableOp!^dense_696/BiasAdd/ReadVariableOp ^dense_696/MatMul/ReadVariableOp0^dense_696/kernel/Regularizer/Abs/ReadVariableOp!^dense_697/BiasAdd/ReadVariableOp ^dense_697/MatMul/ReadVariableOp0^dense_697/kernel/Regularizer/Abs/ReadVariableOp!^dense_698/BiasAdd/ReadVariableOp ^dense_698/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : 2R
'batch_normalization_625/AssignMovingAvg'batch_normalization_625/AssignMovingAvg2p
6batch_normalization_625/AssignMovingAvg/ReadVariableOp6batch_normalization_625/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_625/AssignMovingAvg_1)batch_normalization_625/AssignMovingAvg_12t
8batch_normalization_625/AssignMovingAvg_1/ReadVariableOp8batch_normalization_625/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_625/batchnorm/ReadVariableOp0batch_normalization_625/batchnorm/ReadVariableOp2l
4batch_normalization_625/batchnorm/mul/ReadVariableOp4batch_normalization_625/batchnorm/mul/ReadVariableOp2R
'batch_normalization_626/AssignMovingAvg'batch_normalization_626/AssignMovingAvg2p
6batch_normalization_626/AssignMovingAvg/ReadVariableOp6batch_normalization_626/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_626/AssignMovingAvg_1)batch_normalization_626/AssignMovingAvg_12t
8batch_normalization_626/AssignMovingAvg_1/ReadVariableOp8batch_normalization_626/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_626/batchnorm/ReadVariableOp0batch_normalization_626/batchnorm/ReadVariableOp2l
4batch_normalization_626/batchnorm/mul/ReadVariableOp4batch_normalization_626/batchnorm/mul/ReadVariableOp2R
'batch_normalization_627/AssignMovingAvg'batch_normalization_627/AssignMovingAvg2p
6batch_normalization_627/AssignMovingAvg/ReadVariableOp6batch_normalization_627/AssignMovingAvg/ReadVariableOp2V
)batch_normalization_627/AssignMovingAvg_1)batch_normalization_627/AssignMovingAvg_12t
8batch_normalization_627/AssignMovingAvg_1/ReadVariableOp8batch_normalization_627/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_627/batchnorm/ReadVariableOp0batch_normalization_627/batchnorm/ReadVariableOp2l
4batch_normalization_627/batchnorm/mul/ReadVariableOp4batch_normalization_627/batchnorm/mul/ReadVariableOp2D
 dense_695/BiasAdd/ReadVariableOp dense_695/BiasAdd/ReadVariableOp2B
dense_695/MatMul/ReadVariableOpdense_695/MatMul/ReadVariableOp2b
/dense_695/kernel/Regularizer/Abs/ReadVariableOp/dense_695/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_696/BiasAdd/ReadVariableOp dense_696/BiasAdd/ReadVariableOp2B
dense_696/MatMul/ReadVariableOpdense_696/MatMul/ReadVariableOp2b
/dense_696/kernel/Regularizer/Abs/ReadVariableOp/dense_696/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_697/BiasAdd/ReadVariableOp dense_697/BiasAdd/ReadVariableOp2B
dense_697/MatMul/ReadVariableOpdense_697/MatMul/ReadVariableOp2b
/dense_697/kernel/Regularizer/Abs/ReadVariableOp/dense_697/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_698/BiasAdd/ReadVariableOp dense_698/BiasAdd/ReadVariableOp2B
dense_698/MatMul/ReadVariableOpdense_698/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
­
M
1__inference_leaky_re_lu_627_layer_call_fn_1237967

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
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_627_layer_call_and_return_conditional_losses_1236615`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
¢T
º
J__inference_sequential_70_layer_call_and_return_conditional_losses_1237141
normalization_70_input
normalization_70_sub_y
normalization_70_sqrt_x#
dense_695_1237072:
dense_695_1237074:-
batch_normalization_625_1237077:-
batch_normalization_625_1237079:-
batch_normalization_625_1237081:-
batch_normalization_625_1237083:#
dense_696_1237087:7
dense_696_1237089:7-
batch_normalization_626_1237092:7-
batch_normalization_626_1237094:7-
batch_normalization_626_1237096:7-
batch_normalization_626_1237098:7#
dense_697_1237102:7F
dense_697_1237104:F-
batch_normalization_627_1237107:F-
batch_normalization_627_1237109:F-
batch_normalization_627_1237111:F-
batch_normalization_627_1237113:F#
dense_698_1237117:F
dense_698_1237119:
identity¢/batch_normalization_625/StatefulPartitionedCall¢/batch_normalization_626/StatefulPartitionedCall¢/batch_normalization_627/StatefulPartitionedCall¢!dense_695/StatefulPartitionedCall¢/dense_695/kernel/Regularizer/Abs/ReadVariableOp¢!dense_696/StatefulPartitionedCall¢/dense_696/kernel/Regularizer/Abs/ReadVariableOp¢!dense_697/StatefulPartitionedCall¢/dense_697/kernel/Regularizer/Abs/ReadVariableOp¢!dense_698/StatefulPartitionedCall}
normalization_70/subSubnormalization_70_inputnormalization_70_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_70/SqrtSqrtnormalization_70_sqrt_x*
T0*
_output_shapes

:_
normalization_70/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_70/MaximumMaximumnormalization_70/Sqrt:y:0#normalization_70/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_70/truedivRealDivnormalization_70/sub:z:0normalization_70/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_695/StatefulPartitionedCallStatefulPartitionedCallnormalization_70/truediv:z:0dense_695_1237072dense_695_1237074*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_695_layer_call_and_return_conditional_losses_1236519
/batch_normalization_625/StatefulPartitionedCallStatefulPartitionedCall*dense_695/StatefulPartitionedCall:output:0batch_normalization_625_1237077batch_normalization_625_1237079batch_normalization_625_1237081batch_normalization_625_1237083*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_625_layer_call_and_return_conditional_losses_1236314ù
leaky_re_lu_625/PartitionedCallPartitionedCall8batch_normalization_625/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_625_layer_call_and_return_conditional_losses_1236539
!dense_696/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_625/PartitionedCall:output:0dense_696_1237087dense_696_1237089*
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
F__inference_dense_696_layer_call_and_return_conditional_losses_1236557
/batch_normalization_626/StatefulPartitionedCallStatefulPartitionedCall*dense_696/StatefulPartitionedCall:output:0batch_normalization_626_1237092batch_normalization_626_1237094batch_normalization_626_1237096batch_normalization_626_1237098*
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
T__inference_batch_normalization_626_layer_call_and_return_conditional_losses_1236396ù
leaky_re_lu_626/PartitionedCallPartitionedCall8batch_normalization_626/StatefulPartitionedCall:output:0*
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
L__inference_leaky_re_lu_626_layer_call_and_return_conditional_losses_1236577
!dense_697/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_626/PartitionedCall:output:0dense_697_1237102dense_697_1237104*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_697_layer_call_and_return_conditional_losses_1236595
/batch_normalization_627/StatefulPartitionedCallStatefulPartitionedCall*dense_697/StatefulPartitionedCall:output:0batch_normalization_627_1237107batch_normalization_627_1237109batch_normalization_627_1237111batch_normalization_627_1237113*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_627_layer_call_and_return_conditional_losses_1236478ù
leaky_re_lu_627/PartitionedCallPartitionedCall8batch_normalization_627/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_627_layer_call_and_return_conditional_losses_1236615
!dense_698/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_627/PartitionedCall:output:0dense_698_1237117dense_698_1237119*
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
F__inference_dense_698_layer_call_and_return_conditional_losses_1236627
/dense_695/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_695_1237072*
_output_shapes

:*
dtype0
 dense_695/kernel/Regularizer/AbsAbs7dense_695/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:s
"dense_695/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_695/kernel/Regularizer/SumSum$dense_695/kernel/Regularizer/Abs:y:0+dense_695/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_695/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_695/kernel/Regularizer/mulMul+dense_695/kernel/Regularizer/mul/x:output:0)dense_695/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/dense_696/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_696_1237087*
_output_shapes

:7*
dtype0
 dense_696/kernel/Regularizer/AbsAbs7dense_696/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7s
"dense_696/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_696/kernel/Regularizer/SumSum$dense_696/kernel/Regularizer/Abs:y:0+dense_696/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_696/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_696/kernel/Regularizer/mulMul+dense_696/kernel/Regularizer/mul/x:output:0)dense_696/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/dense_697/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_697_1237102*
_output_shapes

:7F*
dtype0
 dense_697/kernel/Regularizer/AbsAbs7dense_697/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7Fs
"dense_697/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_697/kernel/Regularizer/SumSum$dense_697/kernel/Regularizer/Abs:y:0+dense_697/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_697/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 dense_697/kernel/Regularizer/mulMul+dense_697/kernel/Regularizer/mul/x:output:0)dense_697/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_698/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_625/StatefulPartitionedCall0^batch_normalization_626/StatefulPartitionedCall0^batch_normalization_627/StatefulPartitionedCall"^dense_695/StatefulPartitionedCall0^dense_695/kernel/Regularizer/Abs/ReadVariableOp"^dense_696/StatefulPartitionedCall0^dense_696/kernel/Regularizer/Abs/ReadVariableOp"^dense_697/StatefulPartitionedCall0^dense_697/kernel/Regularizer/Abs/ReadVariableOp"^dense_698/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_625/StatefulPartitionedCall/batch_normalization_625/StatefulPartitionedCall2b
/batch_normalization_626/StatefulPartitionedCall/batch_normalization_626/StatefulPartitionedCall2b
/batch_normalization_627/StatefulPartitionedCall/batch_normalization_627/StatefulPartitionedCall2F
!dense_695/StatefulPartitionedCall!dense_695/StatefulPartitionedCall2b
/dense_695/kernel/Regularizer/Abs/ReadVariableOp/dense_695/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_696/StatefulPartitionedCall!dense_696/StatefulPartitionedCall2b
/dense_696/kernel/Regularizer/Abs/ReadVariableOp/dense_696/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_697/StatefulPartitionedCall!dense_697/StatefulPartitionedCall2b
/dense_697/kernel/Regularizer/Abs/ReadVariableOp/dense_697/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_698/StatefulPartitionedCall!dense_698/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_70_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ñ
³
T__inference_batch_normalization_625_layer_call_and_return_conditional_losses_1236267

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
©
F__inference_dense_695_layer_call_and_return_conditional_losses_1237640

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_695/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/dense_695/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
 dense_695/kernel/Regularizer/AbsAbs7dense_695/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:s
"dense_695/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_695/kernel/Regularizer/SumSum$dense_695/kernel/Regularizer/Abs:y:0+dense_695/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_695/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_695/kernel/Regularizer/mulMul+dense_695/kernel/Regularizer/mul/x:output:0)dense_695/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_695/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_695/kernel/Regularizer/Abs/ReadVariableOp/dense_695/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
h
L__inference_leaky_re_lu_626_layer_call_and_return_conditional_losses_1236577

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
©
®
__inference_loss_fn_0_1238002J
8dense_695_kernel_regularizer_abs_readvariableop_resource:
identity¢/dense_695/kernel/Regularizer/Abs/ReadVariableOp¨
/dense_695/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8dense_695_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:*
dtype0
 dense_695/kernel/Regularizer/AbsAbs7dense_695/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:s
"dense_695/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_695/kernel/Regularizer/SumSum$dense_695/kernel/Regularizer/Abs:y:0+dense_695/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_695/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_695/kernel/Regularizer/mulMul+dense_695/kernel/Regularizer/mul/x:output:0)dense_695/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_695/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^dense_695/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_695/kernel/Regularizer/Abs/ReadVariableOp/dense_695/kernel/Regularizer/Abs/ReadVariableOp
¨T
º
J__inference_sequential_70_layer_call_and_return_conditional_losses_1237062
normalization_70_input
normalization_70_sub_y
normalization_70_sqrt_x#
dense_695_1236993:
dense_695_1236995:-
batch_normalization_625_1236998:-
batch_normalization_625_1237000:-
batch_normalization_625_1237002:-
batch_normalization_625_1237004:#
dense_696_1237008:7
dense_696_1237010:7-
batch_normalization_626_1237013:7-
batch_normalization_626_1237015:7-
batch_normalization_626_1237017:7-
batch_normalization_626_1237019:7#
dense_697_1237023:7F
dense_697_1237025:F-
batch_normalization_627_1237028:F-
batch_normalization_627_1237030:F-
batch_normalization_627_1237032:F-
batch_normalization_627_1237034:F#
dense_698_1237038:F
dense_698_1237040:
identity¢/batch_normalization_625/StatefulPartitionedCall¢/batch_normalization_626/StatefulPartitionedCall¢/batch_normalization_627/StatefulPartitionedCall¢!dense_695/StatefulPartitionedCall¢/dense_695/kernel/Regularizer/Abs/ReadVariableOp¢!dense_696/StatefulPartitionedCall¢/dense_696/kernel/Regularizer/Abs/ReadVariableOp¢!dense_697/StatefulPartitionedCall¢/dense_697/kernel/Regularizer/Abs/ReadVariableOp¢!dense_698/StatefulPartitionedCall}
normalization_70/subSubnormalization_70_inputnormalization_70_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_70/SqrtSqrtnormalization_70_sqrt_x*
T0*
_output_shapes

:_
normalization_70/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_70/MaximumMaximumnormalization_70/Sqrt:y:0#normalization_70/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_70/truedivRealDivnormalization_70/sub:z:0normalization_70/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_695/StatefulPartitionedCallStatefulPartitionedCallnormalization_70/truediv:z:0dense_695_1236993dense_695_1236995*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_695_layer_call_and_return_conditional_losses_1236519
/batch_normalization_625/StatefulPartitionedCallStatefulPartitionedCall*dense_695/StatefulPartitionedCall:output:0batch_normalization_625_1236998batch_normalization_625_1237000batch_normalization_625_1237002batch_normalization_625_1237004*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_625_layer_call_and_return_conditional_losses_1236267ù
leaky_re_lu_625/PartitionedCallPartitionedCall8batch_normalization_625/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_625_layer_call_and_return_conditional_losses_1236539
!dense_696/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_625/PartitionedCall:output:0dense_696_1237008dense_696_1237010*
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
F__inference_dense_696_layer_call_and_return_conditional_losses_1236557
/batch_normalization_626/StatefulPartitionedCallStatefulPartitionedCall*dense_696/StatefulPartitionedCall:output:0batch_normalization_626_1237013batch_normalization_626_1237015batch_normalization_626_1237017batch_normalization_626_1237019*
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
T__inference_batch_normalization_626_layer_call_and_return_conditional_losses_1236349ù
leaky_re_lu_626/PartitionedCallPartitionedCall8batch_normalization_626/StatefulPartitionedCall:output:0*
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
L__inference_leaky_re_lu_626_layer_call_and_return_conditional_losses_1236577
!dense_697/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_626/PartitionedCall:output:0dense_697_1237023dense_697_1237025*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_697_layer_call_and_return_conditional_losses_1236595
/batch_normalization_627/StatefulPartitionedCallStatefulPartitionedCall*dense_697/StatefulPartitionedCall:output:0batch_normalization_627_1237028batch_normalization_627_1237030batch_normalization_627_1237032batch_normalization_627_1237034*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_627_layer_call_and_return_conditional_losses_1236431ù
leaky_re_lu_627/PartitionedCallPartitionedCall8batch_normalization_627/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_627_layer_call_and_return_conditional_losses_1236615
!dense_698/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_627/PartitionedCall:output:0dense_698_1237038dense_698_1237040*
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
F__inference_dense_698_layer_call_and_return_conditional_losses_1236627
/dense_695/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_695_1236993*
_output_shapes

:*
dtype0
 dense_695/kernel/Regularizer/AbsAbs7dense_695/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:s
"dense_695/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_695/kernel/Regularizer/SumSum$dense_695/kernel/Regularizer/Abs:y:0+dense_695/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_695/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_695/kernel/Regularizer/mulMul+dense_695/kernel/Regularizer/mul/x:output:0)dense_695/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/dense_696/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_696_1237008*
_output_shapes

:7*
dtype0
 dense_696/kernel/Regularizer/AbsAbs7dense_696/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7s
"dense_696/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_696/kernel/Regularizer/SumSum$dense_696/kernel/Regularizer/Abs:y:0+dense_696/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_696/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_696/kernel/Regularizer/mulMul+dense_696/kernel/Regularizer/mul/x:output:0)dense_696/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/dense_697/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_697_1237023*
_output_shapes

:7F*
dtype0
 dense_697/kernel/Regularizer/AbsAbs7dense_697/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7Fs
"dense_697/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_697/kernel/Regularizer/SumSum$dense_697/kernel/Regularizer/Abs:y:0+dense_697/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_697/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 dense_697/kernel/Regularizer/mulMul+dense_697/kernel/Regularizer/mul/x:output:0)dense_697/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_698/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_625/StatefulPartitionedCall0^batch_normalization_626/StatefulPartitionedCall0^batch_normalization_627/StatefulPartitionedCall"^dense_695/StatefulPartitionedCall0^dense_695/kernel/Regularizer/Abs/ReadVariableOp"^dense_696/StatefulPartitionedCall0^dense_696/kernel/Regularizer/Abs/ReadVariableOp"^dense_697/StatefulPartitionedCall0^dense_697/kernel/Regularizer/Abs/ReadVariableOp"^dense_698/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_625/StatefulPartitionedCall/batch_normalization_625/StatefulPartitionedCall2b
/batch_normalization_626/StatefulPartitionedCall/batch_normalization_626/StatefulPartitionedCall2b
/batch_normalization_627/StatefulPartitionedCall/batch_normalization_627/StatefulPartitionedCall2F
!dense_695/StatefulPartitionedCall!dense_695/StatefulPartitionedCall2b
/dense_695/kernel/Regularizer/Abs/ReadVariableOp/dense_695/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_696/StatefulPartitionedCall!dense_696/StatefulPartitionedCall2b
/dense_696/kernel/Regularizer/Abs/ReadVariableOp/dense_696/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_697/StatefulPartitionedCall!dense_697/StatefulPartitionedCall2b
/dense_697/kernel/Regularizer/Abs/ReadVariableOp/dense_697/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_698/StatefulPartitionedCall!dense_698/StatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_70_input:$ 

_output_shapes

::$ 

_output_shapes

:
æ
h
L__inference_leaky_re_lu_627_layer_call_and_return_conditional_losses_1236615

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
alpha%>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿF:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Î
©
F__inference_dense_697_layer_call_and_return_conditional_losses_1237882

inputs0
matmul_readvariableop_resource:7F-
biasadd_readvariableop_resource:F
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_697/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7F*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
/dense_697/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:7F*
dtype0
 dense_697/kernel/Regularizer/AbsAbs7dense_697/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7Fs
"dense_697/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_697/kernel/Regularizer/SumSum$dense_697/kernel/Regularizer/Abs:y:0+dense_697/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_697/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 dense_697/kernel/Regularizer/mulMul+dense_697/kernel/Regularizer/mul/x:output:0)dense_697/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF©
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_697/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ7: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_697/kernel/Regularizer/Abs/ReadVariableOp/dense_697/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_625_layer_call_and_return_conditional_losses_1237720

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
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
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_626_layer_call_and_return_conditional_losses_1237807

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
9__inference_batch_normalization_625_layer_call_fn_1237666

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_625_layer_call_and_return_conditional_losses_1236314o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_627_layer_call_and_return_conditional_losses_1236431

inputs/
!batchnorm_readvariableop_resource:F3
%batchnorm_mul_readvariableop_resource:F1
#batchnorm_readvariableop_1_resource:F1
#batchnorm_readvariableop_2_resource:F
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:F*
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
:FP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:F~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Fc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:F*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Fz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:F*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Fr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Ö

/__inference_sequential_70_layer_call_fn_1237212

inputs
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:7
	unknown_8:7
	unknown_9:7

unknown_10:7

unknown_11:7

unknown_12:7

unknown_13:7F

unknown_14:F

unknown_15:F

unknown_16:F

unknown_17:F

unknown_18:F

unknown_19:F

unknown_20:
identity¢StatefulPartitionedCallì
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_70_layer_call_and_return_conditional_losses_1236652o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ñ
³
T__inference_batch_normalization_627_layer_call_and_return_conditional_losses_1237928

inputs/
!batchnorm_readvariableop_resource:F3
%batchnorm_mul_readvariableop_resource:F1
#batchnorm_readvariableop_1_resource:F1
#batchnorm_readvariableop_2_resource:F
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:F*
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
:FP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:F~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Fc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:F*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Fz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:F*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Fr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Æ

+__inference_dense_696_layer_call_fn_1237745

inputs
unknown:7
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
F__inference_dense_696_layer_call_and_return_conditional_losses_1236557o
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
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
í
T__inference_batch_normalization_627_layer_call_and_return_conditional_losses_1237962

inputs5
'assignmovingavg_readvariableop_resource:F7
)assignmovingavg_1_readvariableop_resource:F3
%batchnorm_mul_readvariableop_resource:F/
!batchnorm_readvariableop_resource:F
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:F*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:F
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:F*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:F*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:F*
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
:F*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Fx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:F¬
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
:F*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:F~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:F´
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
:FP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:F~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Fc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Fv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:F*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Fr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿF: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
É	
÷
F__inference_dense_698_layer_call_and_return_conditional_losses_1236627

inputs0
matmul_readvariableop_resource:F-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
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
:ÿÿÿÿÿÿÿÿÿF: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Ñ
³
T__inference_batch_normalization_626_layer_call_and_return_conditional_losses_1236349

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
òS
ª
J__inference_sequential_70_layer_call_and_return_conditional_losses_1236887

inputs
normalization_70_sub_y
normalization_70_sqrt_x#
dense_695_1236818:
dense_695_1236820:-
batch_normalization_625_1236823:-
batch_normalization_625_1236825:-
batch_normalization_625_1236827:-
batch_normalization_625_1236829:#
dense_696_1236833:7
dense_696_1236835:7-
batch_normalization_626_1236838:7-
batch_normalization_626_1236840:7-
batch_normalization_626_1236842:7-
batch_normalization_626_1236844:7#
dense_697_1236848:7F
dense_697_1236850:F-
batch_normalization_627_1236853:F-
batch_normalization_627_1236855:F-
batch_normalization_627_1236857:F-
batch_normalization_627_1236859:F#
dense_698_1236863:F
dense_698_1236865:
identity¢/batch_normalization_625/StatefulPartitionedCall¢/batch_normalization_626/StatefulPartitionedCall¢/batch_normalization_627/StatefulPartitionedCall¢!dense_695/StatefulPartitionedCall¢/dense_695/kernel/Regularizer/Abs/ReadVariableOp¢!dense_696/StatefulPartitionedCall¢/dense_696/kernel/Regularizer/Abs/ReadVariableOp¢!dense_697/StatefulPartitionedCall¢/dense_697/kernel/Regularizer/Abs/ReadVariableOp¢!dense_698/StatefulPartitionedCallm
normalization_70/subSubinputsnormalization_70_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
normalization_70/SqrtSqrtnormalization_70_sqrt_x*
T0*
_output_shapes

:_
normalization_70/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization_70/MaximumMaximumnormalization_70/Sqrt:y:0#normalization_70/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_70/truedivRealDivnormalization_70/sub:z:0normalization_70/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_695/StatefulPartitionedCallStatefulPartitionedCallnormalization_70/truediv:z:0dense_695_1236818dense_695_1236820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_695_layer_call_and_return_conditional_losses_1236519
/batch_normalization_625/StatefulPartitionedCallStatefulPartitionedCall*dense_695/StatefulPartitionedCall:output:0batch_normalization_625_1236823batch_normalization_625_1236825batch_normalization_625_1236827batch_normalization_625_1236829*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_625_layer_call_and_return_conditional_losses_1236314ù
leaky_re_lu_625/PartitionedCallPartitionedCall8batch_normalization_625/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_625_layer_call_and_return_conditional_losses_1236539
!dense_696/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_625/PartitionedCall:output:0dense_696_1236833dense_696_1236835*
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
F__inference_dense_696_layer_call_and_return_conditional_losses_1236557
/batch_normalization_626/StatefulPartitionedCallStatefulPartitionedCall*dense_696/StatefulPartitionedCall:output:0batch_normalization_626_1236838batch_normalization_626_1236840batch_normalization_626_1236842batch_normalization_626_1236844*
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
T__inference_batch_normalization_626_layer_call_and_return_conditional_losses_1236396ù
leaky_re_lu_626/PartitionedCallPartitionedCall8batch_normalization_626/StatefulPartitionedCall:output:0*
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
L__inference_leaky_re_lu_626_layer_call_and_return_conditional_losses_1236577
!dense_697/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_626/PartitionedCall:output:0dense_697_1236848dense_697_1236850*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_697_layer_call_and_return_conditional_losses_1236595
/batch_normalization_627/StatefulPartitionedCallStatefulPartitionedCall*dense_697/StatefulPartitionedCall:output:0batch_normalization_627_1236853batch_normalization_627_1236855batch_normalization_627_1236857batch_normalization_627_1236859*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_batch_normalization_627_layer_call_and_return_conditional_losses_1236478ù
leaky_re_lu_627/PartitionedCallPartitionedCall8batch_normalization_627/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_leaky_re_lu_627_layer_call_and_return_conditional_losses_1236615
!dense_698/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_627/PartitionedCall:output:0dense_698_1236863dense_698_1236865*
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
F__inference_dense_698_layer_call_and_return_conditional_losses_1236627
/dense_695/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_695_1236818*
_output_shapes

:*
dtype0
 dense_695/kernel/Regularizer/AbsAbs7dense_695/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:s
"dense_695/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_695/kernel/Regularizer/SumSum$dense_695/kernel/Regularizer/Abs:y:0+dense_695/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_695/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_695/kernel/Regularizer/mulMul+dense_695/kernel/Regularizer/mul/x:output:0)dense_695/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/dense_696/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_696_1236833*
_output_shapes

:7*
dtype0
 dense_696/kernel/Regularizer/AbsAbs7dense_696/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7s
"dense_696/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_696/kernel/Regularizer/SumSum$dense_696/kernel/Regularizer/Abs:y:0+dense_696/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_696/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ= 
 dense_696/kernel/Regularizer/mulMul+dense_696/kernel/Regularizer/mul/x:output:0)dense_696/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/dense_697/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_697_1236848*
_output_shapes

:7F*
dtype0
 dense_697/kernel/Regularizer/AbsAbs7dense_697/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:7Fs
"dense_697/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_697/kernel/Regularizer/SumSum$dense_697/kernel/Regularizer/Abs:y:0+dense_697/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_697/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *·Ñ8 
 dense_697/kernel/Regularizer/mulMul+dense_697/kernel/Regularizer/mul/x:output:0)dense_697/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_698/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp0^batch_normalization_625/StatefulPartitionedCall0^batch_normalization_626/StatefulPartitionedCall0^batch_normalization_627/StatefulPartitionedCall"^dense_695/StatefulPartitionedCall0^dense_695/kernel/Regularizer/Abs/ReadVariableOp"^dense_696/StatefulPartitionedCall0^dense_696/kernel/Regularizer/Abs/ReadVariableOp"^dense_697/StatefulPartitionedCall0^dense_697/kernel/Regularizer/Abs/ReadVariableOp"^dense_698/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_625/StatefulPartitionedCall/batch_normalization_625/StatefulPartitionedCall2b
/batch_normalization_626/StatefulPartitionedCall/batch_normalization_626/StatefulPartitionedCall2b
/batch_normalization_627/StatefulPartitionedCall/batch_normalization_627/StatefulPartitionedCall2F
!dense_695/StatefulPartitionedCall!dense_695/StatefulPartitionedCall2b
/dense_695/kernel/Regularizer/Abs/ReadVariableOp/dense_695/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_696/StatefulPartitionedCall!dense_696/StatefulPartitionedCall2b
/dense_696/kernel/Regularizer/Abs/ReadVariableOp/dense_696/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_697/StatefulPartitionedCall!dense_697/StatefulPartitionedCall2b
/dense_697/kernel/Regularizer/Abs/ReadVariableOp/dense_697/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_698/StatefulPartitionedCall!dense_698/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Æ

+__inference_dense_698_layer_call_fn_1237981

inputs
unknown:F
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
F__inference_dense_698_layer_call_and_return_conditional_losses_1236627o
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
:ÿÿÿÿÿÿÿÿÿF: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
 
_user_specified_nameinputs
Ô

%__inference_signature_wrapper_1237562
normalization_70_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:7
	unknown_8:7
	unknown_9:7

unknown_10:7

unknown_11:7

unknown_12:7

unknown_13:7F

unknown_14:F

unknown_15:F

unknown_16:F

unknown_17:F

unknown_18:F

unknown_19:F

unknown_20:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallnormalization_70_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1236243o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_70_input:$ 

_output_shapes

::$ 

_output_shapes

:

¾
"__inference__wrapped_model_1236243
normalization_70_input(
$sequential_70_normalization_70_sub_y)
%sequential_70_normalization_70_sqrt_xH
6sequential_70_dense_695_matmul_readvariableop_resource:E
7sequential_70_dense_695_biasadd_readvariableop_resource:U
Gsequential_70_batch_normalization_625_batchnorm_readvariableop_resource:Y
Ksequential_70_batch_normalization_625_batchnorm_mul_readvariableop_resource:W
Isequential_70_batch_normalization_625_batchnorm_readvariableop_1_resource:W
Isequential_70_batch_normalization_625_batchnorm_readvariableop_2_resource:H
6sequential_70_dense_696_matmul_readvariableop_resource:7E
7sequential_70_dense_696_biasadd_readvariableop_resource:7U
Gsequential_70_batch_normalization_626_batchnorm_readvariableop_resource:7Y
Ksequential_70_batch_normalization_626_batchnorm_mul_readvariableop_resource:7W
Isequential_70_batch_normalization_626_batchnorm_readvariableop_1_resource:7W
Isequential_70_batch_normalization_626_batchnorm_readvariableop_2_resource:7H
6sequential_70_dense_697_matmul_readvariableop_resource:7FE
7sequential_70_dense_697_biasadd_readvariableop_resource:FU
Gsequential_70_batch_normalization_627_batchnorm_readvariableop_resource:FY
Ksequential_70_batch_normalization_627_batchnorm_mul_readvariableop_resource:FW
Isequential_70_batch_normalization_627_batchnorm_readvariableop_1_resource:FW
Isequential_70_batch_normalization_627_batchnorm_readvariableop_2_resource:FH
6sequential_70_dense_698_matmul_readvariableop_resource:FE
7sequential_70_dense_698_biasadd_readvariableop_resource:
identity¢>sequential_70/batch_normalization_625/batchnorm/ReadVariableOp¢@sequential_70/batch_normalization_625/batchnorm/ReadVariableOp_1¢@sequential_70/batch_normalization_625/batchnorm/ReadVariableOp_2¢Bsequential_70/batch_normalization_625/batchnorm/mul/ReadVariableOp¢>sequential_70/batch_normalization_626/batchnorm/ReadVariableOp¢@sequential_70/batch_normalization_626/batchnorm/ReadVariableOp_1¢@sequential_70/batch_normalization_626/batchnorm/ReadVariableOp_2¢Bsequential_70/batch_normalization_626/batchnorm/mul/ReadVariableOp¢>sequential_70/batch_normalization_627/batchnorm/ReadVariableOp¢@sequential_70/batch_normalization_627/batchnorm/ReadVariableOp_1¢@sequential_70/batch_normalization_627/batchnorm/ReadVariableOp_2¢Bsequential_70/batch_normalization_627/batchnorm/mul/ReadVariableOp¢.sequential_70/dense_695/BiasAdd/ReadVariableOp¢-sequential_70/dense_695/MatMul/ReadVariableOp¢.sequential_70/dense_696/BiasAdd/ReadVariableOp¢-sequential_70/dense_696/MatMul/ReadVariableOp¢.sequential_70/dense_697/BiasAdd/ReadVariableOp¢-sequential_70/dense_697/MatMul/ReadVariableOp¢.sequential_70/dense_698/BiasAdd/ReadVariableOp¢-sequential_70/dense_698/MatMul/ReadVariableOp
"sequential_70/normalization_70/subSubnormalization_70_input$sequential_70_normalization_70_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
#sequential_70/normalization_70/SqrtSqrt%sequential_70_normalization_70_sqrt_x*
T0*
_output_shapes

:m
(sequential_70/normalization_70/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3¶
&sequential_70/normalization_70/MaximumMaximum'sequential_70/normalization_70/Sqrt:y:01sequential_70/normalization_70/Maximum/y:output:0*
T0*
_output_shapes

:·
&sequential_70/normalization_70/truedivRealDiv&sequential_70/normalization_70/sub:z:0*sequential_70/normalization_70/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_70/dense_695/MatMul/ReadVariableOpReadVariableOp6sequential_70_dense_695_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_70/dense_695/MatMulMatMul*sequential_70/normalization_70/truediv:z:05sequential_70/dense_695/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_70/dense_695/BiasAdd/ReadVariableOpReadVariableOp7sequential_70_dense_695_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_70/dense_695/BiasAddBiasAdd(sequential_70/dense_695/MatMul:product:06sequential_70/dense_695/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
>sequential_70/batch_normalization_625/batchnorm/ReadVariableOpReadVariableOpGsequential_70_batch_normalization_625_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0z
5sequential_70/batch_normalization_625/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_70/batch_normalization_625/batchnorm/addAddV2Fsequential_70/batch_normalization_625/batchnorm/ReadVariableOp:value:0>sequential_70/batch_normalization_625/batchnorm/add/y:output:0*
T0*
_output_shapes
:
5sequential_70/batch_normalization_625/batchnorm/RsqrtRsqrt7sequential_70/batch_normalization_625/batchnorm/add:z:0*
T0*
_output_shapes
:Ê
Bsequential_70/batch_normalization_625/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_70_batch_normalization_625_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0æ
3sequential_70/batch_normalization_625/batchnorm/mulMul9sequential_70/batch_normalization_625/batchnorm/Rsqrt:y:0Jsequential_70/batch_normalization_625/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ñ
5sequential_70/batch_normalization_625/batchnorm/mul_1Mul(sequential_70/dense_695/BiasAdd:output:07sequential_70/batch_normalization_625/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_70/batch_normalization_625/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_70_batch_normalization_625_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ä
5sequential_70/batch_normalization_625/batchnorm/mul_2MulHsequential_70/batch_normalization_625/batchnorm/ReadVariableOp_1:value:07sequential_70/batch_normalization_625/batchnorm/mul:z:0*
T0*
_output_shapes
:Æ
@sequential_70/batch_normalization_625/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_70_batch_normalization_625_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ä
3sequential_70/batch_normalization_625/batchnorm/subSubHsequential_70/batch_normalization_625/batchnorm/ReadVariableOp_2:value:09sequential_70/batch_normalization_625/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ä
5sequential_70/batch_normalization_625/batchnorm/add_1AddV29sequential_70/batch_normalization_625/batchnorm/mul_1:z:07sequential_70/batch_normalization_625/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
'sequential_70/leaky_re_lu_625/LeakyRelu	LeakyRelu9sequential_70/batch_normalization_625/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
alpha%>¤
-sequential_70/dense_696/MatMul/ReadVariableOpReadVariableOp6sequential_70_dense_696_matmul_readvariableop_resource*
_output_shapes

:7*
dtype0È
sequential_70/dense_696/MatMulMatMul5sequential_70/leaky_re_lu_625/LeakyRelu:activations:05sequential_70/dense_696/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¢
.sequential_70/dense_696/BiasAdd/ReadVariableOpReadVariableOp7sequential_70_dense_696_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0¾
sequential_70/dense_696/BiasAddBiasAdd(sequential_70/dense_696/MatMul:product:06sequential_70/dense_696/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7Â
>sequential_70/batch_normalization_626/batchnorm/ReadVariableOpReadVariableOpGsequential_70_batch_normalization_626_batchnorm_readvariableop_resource*
_output_shapes
:7*
dtype0z
5sequential_70/batch_normalization_626/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_70/batch_normalization_626/batchnorm/addAddV2Fsequential_70/batch_normalization_626/batchnorm/ReadVariableOp:value:0>sequential_70/batch_normalization_626/batchnorm/add/y:output:0*
T0*
_output_shapes
:7
5sequential_70/batch_normalization_626/batchnorm/RsqrtRsqrt7sequential_70/batch_normalization_626/batchnorm/add:z:0*
T0*
_output_shapes
:7Ê
Bsequential_70/batch_normalization_626/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_70_batch_normalization_626_batchnorm_mul_readvariableop_resource*
_output_shapes
:7*
dtype0æ
3sequential_70/batch_normalization_626/batchnorm/mulMul9sequential_70/batch_normalization_626/batchnorm/Rsqrt:y:0Jsequential_70/batch_normalization_626/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:7Ñ
5sequential_70/batch_normalization_626/batchnorm/mul_1Mul(sequential_70/dense_696/BiasAdd:output:07sequential_70/batch_normalization_626/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7Æ
@sequential_70/batch_normalization_626/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_70_batch_normalization_626_batchnorm_readvariableop_1_resource*
_output_shapes
:7*
dtype0ä
5sequential_70/batch_normalization_626/batchnorm/mul_2MulHsequential_70/batch_normalization_626/batchnorm/ReadVariableOp_1:value:07sequential_70/batch_normalization_626/batchnorm/mul:z:0*
T0*
_output_shapes
:7Æ
@sequential_70/batch_normalization_626/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_70_batch_normalization_626_batchnorm_readvariableop_2_resource*
_output_shapes
:7*
dtype0ä
3sequential_70/batch_normalization_626/batchnorm/subSubHsequential_70/batch_normalization_626/batchnorm/ReadVariableOp_2:value:09sequential_70/batch_normalization_626/batchnorm/mul_2:z:0*
T0*
_output_shapes
:7ä
5sequential_70/batch_normalization_626/batchnorm/add_1AddV29sequential_70/batch_normalization_626/batchnorm/mul_1:z:07sequential_70/batch_normalization_626/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7¨
'sequential_70/leaky_re_lu_626/LeakyRelu	LeakyRelu9sequential_70/batch_normalization_626/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ7*
alpha%>¤
-sequential_70/dense_697/MatMul/ReadVariableOpReadVariableOp6sequential_70_dense_697_matmul_readvariableop_resource*
_output_shapes

:7F*
dtype0È
sequential_70/dense_697/MatMulMatMul5sequential_70/leaky_re_lu_626/LeakyRelu:activations:05sequential_70/dense_697/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF¢
.sequential_70/dense_697/BiasAdd/ReadVariableOpReadVariableOp7sequential_70_dense_697_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype0¾
sequential_70/dense_697/BiasAddBiasAdd(sequential_70/dense_697/MatMul:product:06sequential_70/dense_697/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFÂ
>sequential_70/batch_normalization_627/batchnorm/ReadVariableOpReadVariableOpGsequential_70_batch_normalization_627_batchnorm_readvariableop_resource*
_output_shapes
:F*
dtype0z
5sequential_70/batch_normalization_627/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
3sequential_70/batch_normalization_627/batchnorm/addAddV2Fsequential_70/batch_normalization_627/batchnorm/ReadVariableOp:value:0>sequential_70/batch_normalization_627/batchnorm/add/y:output:0*
T0*
_output_shapes
:F
5sequential_70/batch_normalization_627/batchnorm/RsqrtRsqrt7sequential_70/batch_normalization_627/batchnorm/add:z:0*
T0*
_output_shapes
:FÊ
Bsequential_70/batch_normalization_627/batchnorm/mul/ReadVariableOpReadVariableOpKsequential_70_batch_normalization_627_batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype0æ
3sequential_70/batch_normalization_627/batchnorm/mulMul9sequential_70/batch_normalization_627/batchnorm/Rsqrt:y:0Jsequential_70/batch_normalization_627/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:FÑ
5sequential_70/batch_normalization_627/batchnorm/mul_1Mul(sequential_70/dense_697/BiasAdd:output:07sequential_70/batch_normalization_627/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿFÆ
@sequential_70/batch_normalization_627/batchnorm/ReadVariableOp_1ReadVariableOpIsequential_70_batch_normalization_627_batchnorm_readvariableop_1_resource*
_output_shapes
:F*
dtype0ä
5sequential_70/batch_normalization_627/batchnorm/mul_2MulHsequential_70/batch_normalization_627/batchnorm/ReadVariableOp_1:value:07sequential_70/batch_normalization_627/batchnorm/mul:z:0*
T0*
_output_shapes
:FÆ
@sequential_70/batch_normalization_627/batchnorm/ReadVariableOp_2ReadVariableOpIsequential_70_batch_normalization_627_batchnorm_readvariableop_2_resource*
_output_shapes
:F*
dtype0ä
3sequential_70/batch_normalization_627/batchnorm/subSubHsequential_70/batch_normalization_627/batchnorm/ReadVariableOp_2:value:09sequential_70/batch_normalization_627/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Fä
5sequential_70/batch_normalization_627/batchnorm/add_1AddV29sequential_70/batch_normalization_627/batchnorm/mul_1:z:07sequential_70/batch_normalization_627/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF¨
'sequential_70/leaky_re_lu_627/LeakyRelu	LeakyRelu9sequential_70/batch_normalization_627/batchnorm/add_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF*
alpha%>¤
-sequential_70/dense_698/MatMul/ReadVariableOpReadVariableOp6sequential_70_dense_698_matmul_readvariableop_resource*
_output_shapes

:F*
dtype0È
sequential_70/dense_698/MatMulMatMul5sequential_70/leaky_re_lu_627/LeakyRelu:activations:05sequential_70/dense_698/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_70/dense_698/BiasAdd/ReadVariableOpReadVariableOp7sequential_70_dense_698_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_70/dense_698/BiasAddBiasAdd(sequential_70/dense_698/MatMul:product:06sequential_70/dense_698/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_70/dense_698/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿî	
NoOpNoOp?^sequential_70/batch_normalization_625/batchnorm/ReadVariableOpA^sequential_70/batch_normalization_625/batchnorm/ReadVariableOp_1A^sequential_70/batch_normalization_625/batchnorm/ReadVariableOp_2C^sequential_70/batch_normalization_625/batchnorm/mul/ReadVariableOp?^sequential_70/batch_normalization_626/batchnorm/ReadVariableOpA^sequential_70/batch_normalization_626/batchnorm/ReadVariableOp_1A^sequential_70/batch_normalization_626/batchnorm/ReadVariableOp_2C^sequential_70/batch_normalization_626/batchnorm/mul/ReadVariableOp?^sequential_70/batch_normalization_627/batchnorm/ReadVariableOpA^sequential_70/batch_normalization_627/batchnorm/ReadVariableOp_1A^sequential_70/batch_normalization_627/batchnorm/ReadVariableOp_2C^sequential_70/batch_normalization_627/batchnorm/mul/ReadVariableOp/^sequential_70/dense_695/BiasAdd/ReadVariableOp.^sequential_70/dense_695/MatMul/ReadVariableOp/^sequential_70/dense_696/BiasAdd/ReadVariableOp.^sequential_70/dense_696/MatMul/ReadVariableOp/^sequential_70/dense_697/BiasAdd/ReadVariableOp.^sequential_70/dense_697/MatMul/ReadVariableOp/^sequential_70/dense_698/BiasAdd/ReadVariableOp.^sequential_70/dense_698/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : : : : : : : 2
>sequential_70/batch_normalization_625/batchnorm/ReadVariableOp>sequential_70/batch_normalization_625/batchnorm/ReadVariableOp2
@sequential_70/batch_normalization_625/batchnorm/ReadVariableOp_1@sequential_70/batch_normalization_625/batchnorm/ReadVariableOp_12
@sequential_70/batch_normalization_625/batchnorm/ReadVariableOp_2@sequential_70/batch_normalization_625/batchnorm/ReadVariableOp_22
Bsequential_70/batch_normalization_625/batchnorm/mul/ReadVariableOpBsequential_70/batch_normalization_625/batchnorm/mul/ReadVariableOp2
>sequential_70/batch_normalization_626/batchnorm/ReadVariableOp>sequential_70/batch_normalization_626/batchnorm/ReadVariableOp2
@sequential_70/batch_normalization_626/batchnorm/ReadVariableOp_1@sequential_70/batch_normalization_626/batchnorm/ReadVariableOp_12
@sequential_70/batch_normalization_626/batchnorm/ReadVariableOp_2@sequential_70/batch_normalization_626/batchnorm/ReadVariableOp_22
Bsequential_70/batch_normalization_626/batchnorm/mul/ReadVariableOpBsequential_70/batch_normalization_626/batchnorm/mul/ReadVariableOp2
>sequential_70/batch_normalization_627/batchnorm/ReadVariableOp>sequential_70/batch_normalization_627/batchnorm/ReadVariableOp2
@sequential_70/batch_normalization_627/batchnorm/ReadVariableOp_1@sequential_70/batch_normalization_627/batchnorm/ReadVariableOp_12
@sequential_70/batch_normalization_627/batchnorm/ReadVariableOp_2@sequential_70/batch_normalization_627/batchnorm/ReadVariableOp_22
Bsequential_70/batch_normalization_627/batchnorm/mul/ReadVariableOpBsequential_70/batch_normalization_627/batchnorm/mul/ReadVariableOp2`
.sequential_70/dense_695/BiasAdd/ReadVariableOp.sequential_70/dense_695/BiasAdd/ReadVariableOp2^
-sequential_70/dense_695/MatMul/ReadVariableOp-sequential_70/dense_695/MatMul/ReadVariableOp2`
.sequential_70/dense_696/BiasAdd/ReadVariableOp.sequential_70/dense_696/BiasAdd/ReadVariableOp2^
-sequential_70/dense_696/MatMul/ReadVariableOp-sequential_70/dense_696/MatMul/ReadVariableOp2`
.sequential_70/dense_697/BiasAdd/ReadVariableOp.sequential_70/dense_697/BiasAdd/ReadVariableOp2^
-sequential_70/dense_697/MatMul/ReadVariableOp-sequential_70/dense_697/MatMul/ReadVariableOp2`
.sequential_70/dense_698/BiasAdd/ReadVariableOp.sequential_70/dense_698/BiasAdd/ReadVariableOp2^
-sequential_70/dense_698/MatMul/ReadVariableOp-sequential_70/dense_698/MatMul/ReadVariableOp:_ [
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0
_user_specified_namenormalization_70_input:$ 

_output_shapes

::$ 

_output_shapes

:
¬
Ô
9__inference_batch_normalization_626_layer_call_fn_1237787

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
T__inference_batch_normalization_626_layer_call_and_return_conditional_losses_1236396o
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
1__inference_leaky_re_lu_626_layer_call_fn_1237846

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
L__inference_leaky_re_lu_626_layer_call_and_return_conditional_losses_1236577`
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
normalization_70_input?
(serving_default_normalization_70_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_6980
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ñ
Æ
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
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ó

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
	keras_api
_adapt_function"
_tf_keras_layer
»

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
»

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Pkernel
Qbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
Xaxis
	Ygamma
Zbeta
[moving_mean
\moving_variance
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
»

ikernel
jbias
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses"
_tf_keras_layer
Ø
qiter

rbeta_1

sbeta_2
	tdecaymµm¶'m·(m¸7m¹8mº@m»Am¼Pm½Qm¾Ym¿ZmÀimÁjmÂvÃvÄ'vÅ(vÆ7vÇ8vÈ@vÉAvÊPvËQvÌYvÍZvÎivÏjvÐ"
	optimizer
Î
0
1
2
3
4
'5
(6
)7
*8
79
810
@11
A12
B13
C14
P15
Q16
Y17
Z18
[19
\20
i21
j22"
trackable_list_wrapper

0
1
'2
(3
74
85
@6
A7
P8
Q9
Y10
Z11
i12
j13"
trackable_list_wrapper
5
u0
v1
w2"
trackable_list_wrapper
Ê
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
/__inference_sequential_70_layer_call_fn_1236699
/__inference_sequential_70_layer_call_fn_1237212
/__inference_sequential_70_layer_call_fn_1237261
/__inference_sequential_70_layer_call_fn_1236983À
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
J__inference_sequential_70_layer_call_and_return_conditional_losses_1237365
J__inference_sequential_70_layer_call_and_return_conditional_losses_1237511
J__inference_sequential_70_layer_call_and_return_conditional_losses_1237062
J__inference_sequential_70_layer_call_and_return_conditional_losses_1237141À
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
"__inference__wrapped_model_1236243normalization_70_input"
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
,
}serving_default"
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
À2½
__inference_adapt_step_1237609
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
": 2dense_695/kernel
:2dense_695/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
u0"
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_695_layer_call_fn_1237624¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
F__inference_dense_695_layer_call_and_return_conditional_losses_1237640¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)2batch_normalization_625/gamma
*:(2batch_normalization_625/beta
3:1 (2#batch_normalization_625/moving_mean
7:5 (2'batch_normalization_625/moving_variance
<
'0
(1
)2
*3"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_625_layer_call_fn_1237653
9__inference_batch_normalization_625_layer_call_fn_1237666´
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
T__inference_batch_normalization_625_layer_call_and_return_conditional_losses_1237686
T__inference_batch_normalization_625_layer_call_and_return_conditional_losses_1237720´
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_625_layer_call_fn_1237725¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
L__inference_leaky_re_lu_625_layer_call_and_return_conditional_losses_1237730¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": 72dense_696/kernel
:72dense_696/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
'
v0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_696_layer_call_fn_1237745¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
F__inference_dense_696_layer_call_and_return_conditional_losses_1237761¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)72batch_normalization_626/gamma
*:(72batch_normalization_626/beta
3:17 (2#batch_normalization_626/moving_mean
7:57 (2'batch_normalization_626/moving_variance
<
@0
A1
B2
C3"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_626_layer_call_fn_1237774
9__inference_batch_normalization_626_layer_call_fn_1237787´
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
T__inference_batch_normalization_626_layer_call_and_return_conditional_losses_1237807
T__inference_batch_normalization_626_layer_call_and_return_conditional_losses_1237841´
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_626_layer_call_fn_1237846¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
L__inference_leaky_re_lu_626_layer_call_and_return_conditional_losses_1237851¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": 7F2dense_697/kernel
:F2dense_697/bias
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
'
w0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_697_layer_call_fn_1237866¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
F__inference_dense_697_layer_call_and_return_conditional_losses_1237882¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
+:)F2batch_normalization_627/gamma
*:(F2batch_normalization_627/beta
3:1F (2#batch_normalization_627/moving_mean
7:5F (2'batch_normalization_627/moving_variance
<
Y0
Z1
[2
\3"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
°2­
9__inference_batch_normalization_627_layer_call_fn_1237895
9__inference_batch_normalization_627_layer_call_fn_1237908´
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
T__inference_batch_normalization_627_layer_call_and_return_conditional_losses_1237928
T__inference_batch_normalization_627_layer_call_and_return_conditional_losses_1237962´
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
¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_leaky_re_lu_627_layer_call_fn_1237967¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
L__inference_leaky_re_lu_627_layer_call_and_return_conditional_losses_1237972¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": F2dense_698/kernel
:2dense_698/bias
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_698_layer_call_fn_1237981¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
F__inference_dense_698_layer_call_and_return_conditional_losses_1237991¢
²
FullArgSpec
args
jself
jinputs
varargs
 
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
__inference_loss_fn_0_1238002
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
__inference_loss_fn_1_1238013
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
__inference_loss_fn_2_1238024
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
_
0
1
2
)3
*4
B5
C6
[7
\8"
trackable_list_wrapper
n
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
10"
trackable_list_wrapper
(
°0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
%__inference_signature_wrapper_1237562normalization_70_input"
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
'
u0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
)0
*1"
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
'
v0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
B0
C1"
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
'
w0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
[0
\1"
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

±total

²count
³	variables
´	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
±0
²1"
trackable_list_wrapper
.
³	variables"
_generic_user_object
':%2Adam/dense_695/kernel/m
!:2Adam/dense_695/bias/m
0:.2$Adam/batch_normalization_625/gamma/m
/:-2#Adam/batch_normalization_625/beta/m
':%72Adam/dense_696/kernel/m
!:72Adam/dense_696/bias/m
0:.72$Adam/batch_normalization_626/gamma/m
/:-72#Adam/batch_normalization_626/beta/m
':%7F2Adam/dense_697/kernel/m
!:F2Adam/dense_697/bias/m
0:.F2$Adam/batch_normalization_627/gamma/m
/:-F2#Adam/batch_normalization_627/beta/m
':%F2Adam/dense_698/kernel/m
!:2Adam/dense_698/bias/m
':%2Adam/dense_695/kernel/v
!:2Adam/dense_695/bias/v
0:.2$Adam/batch_normalization_625/gamma/v
/:-2#Adam/batch_normalization_625/beta/v
':%72Adam/dense_696/kernel/v
!:72Adam/dense_696/bias/v
0:.72$Adam/batch_normalization_626/gamma/v
/:-72#Adam/batch_normalization_626/beta/v
':%7F2Adam/dense_697/kernel/v
!:F2Adam/dense_697/bias/v
0:.F2$Adam/batch_normalization_627/gamma/v
/:-F2#Adam/batch_normalization_627/beta/v
':%F2Adam/dense_698/kernel/v
!:2Adam/dense_698/bias/v
	J
Const
J	
Const_1¹
"__inference__wrapped_model_1236243ÑÒ*')(78C@BAPQ\Y[Zij?¢<
5¢2
0-
normalization_70_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_698# 
	dense_698ÿÿÿÿÿÿÿÿÿp
__inference_adapt_step_1237609NC¢@
9¢6
41¢
ÿÿÿÿÿÿÿÿÿIteratorSpec 
ª "
 º
T__inference_batch_normalization_625_layer_call_and_return_conditional_losses_1237686b*')(3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
T__inference_batch_normalization_625_layer_call_and_return_conditional_losses_1237720b)*'(3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
9__inference_batch_normalization_625_layer_call_fn_1237653U*')(3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
9__inference_batch_normalization_625_layer_call_fn_1237666U)*'(3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿº
T__inference_batch_normalization_626_layer_call_and_return_conditional_losses_1237807bC@BA3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 º
T__inference_batch_normalization_626_layer_call_and_return_conditional_losses_1237841bBC@A3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 
9__inference_batch_normalization_626_layer_call_fn_1237774UC@BA3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p 
ª "ÿÿÿÿÿÿÿÿÿ7
9__inference_batch_normalization_626_layer_call_fn_1237787UBC@A3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ7
p
ª "ÿÿÿÿÿÿÿÿÿ7º
T__inference_batch_normalization_627_layer_call_and_return_conditional_losses_1237928b\Y[Z3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿF
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 º
T__inference_batch_normalization_627_layer_call_and_return_conditional_losses_1237962b[\YZ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿF
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 
9__inference_batch_normalization_627_layer_call_fn_1237895U\Y[Z3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿF
p 
ª "ÿÿÿÿÿÿÿÿÿF
9__inference_batch_normalization_627_layer_call_fn_1237908U[\YZ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿF
p
ª "ÿÿÿÿÿÿÿÿÿF¦
F__inference_dense_695_layer_call_and_return_conditional_losses_1237640\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_695_layer_call_fn_1237624O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_696_layer_call_and_return_conditional_losses_1237761\78/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 ~
+__inference_dense_696_layer_call_fn_1237745O78/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ7¦
F__inference_dense_697_layer_call_and_return_conditional_losses_1237882\PQ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 ~
+__inference_dense_697_layer_call_fn_1237866OPQ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "ÿÿÿÿÿÿÿÿÿF¦
F__inference_dense_698_layer_call_and_return_conditional_losses_1237991\ij/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿF
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_698_layer_call_fn_1237981Oij/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿF
ª "ÿÿÿÿÿÿÿÿÿ¨
L__inference_leaky_re_lu_625_layer_call_and_return_conditional_losses_1237730X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_leaky_re_lu_625_layer_call_fn_1237725K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
L__inference_leaky_re_lu_626_layer_call_and_return_conditional_losses_1237851X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ7
 
1__inference_leaky_re_lu_626_layer_call_fn_1237846K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ7
ª "ÿÿÿÿÿÿÿÿÿ7¨
L__inference_leaky_re_lu_627_layer_call_and_return_conditional_losses_1237972X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿF
ª "%¢"

0ÿÿÿÿÿÿÿÿÿF
 
1__inference_leaky_re_lu_627_layer_call_fn_1237967K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿF
ª "ÿÿÿÿÿÿÿÿÿF<
__inference_loss_fn_0_1238002¢

¢ 
ª " <
__inference_loss_fn_1_12380137¢

¢ 
ª " <
__inference_loss_fn_2_1238024P¢

¢ 
ª " Ù
J__inference_sequential_70_layer_call_and_return_conditional_losses_1237062ÑÒ*')(78C@BAPQ\Y[ZijG¢D
=¢:
0-
normalization_70_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ù
J__inference_sequential_70_layer_call_and_return_conditional_losses_1237141ÑÒ)*'(78BC@APQ[\YZijG¢D
=¢:
0-
normalization_70_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 È
J__inference_sequential_70_layer_call_and_return_conditional_losses_1237365zÑÒ*')(78C@BAPQ\Y[Zij7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 È
J__inference_sequential_70_layer_call_and_return_conditional_losses_1237511zÑÒ)*'(78BC@APQ[\YZij7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
/__inference_sequential_70_layer_call_fn_1236699}ÑÒ*')(78C@BAPQ\Y[ZijG¢D
=¢:
0-
normalization_70_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ°
/__inference_sequential_70_layer_call_fn_1236983}ÑÒ)*'(78BC@APQ[\YZijG¢D
=¢:
0-
normalization_70_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
/__inference_sequential_70_layer_call_fn_1237212mÑÒ*')(78C@BAPQ\Y[Zij7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
/__inference_sequential_70_layer_call_fn_1237261mÑÒ)*'(78BC@APQ[\YZij7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÖ
%__inference_signature_wrapper_1237562¬ÑÒ*')(78C@BAPQ\Y[ZijY¢V
¢ 
OªL
J
normalization_70_input0-
normalization_70_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_698# 
	dense_698ÿÿÿÿÿÿÿÿÿ