       £K"	   1a†÷Abrain.Event:2°_В±5      ўOо	e/1a†÷A"•k
p
PlaceholderPlaceholder*
dtype0*
shape:€€€€€€€€€Р*(
_output_shapes
:€€€€€€€€€Р
`
	out/zerosConst*
valueB	Р
*    *
dtype0*
_output_shapes
:	Р

{
out/W
VariableV2*
shape:	Р
*
dtype0*
	container *
shared_name *
_output_shapes
:	Р

Х
out/W/AssignAssignout/W	out/zeros*
T0*
validate_shape(*
use_locking(*
_class

loc:@out/W*
_output_shapes
:	Р

a

out/W/readIdentityout/W*
T0*
_class

loc:@out/W*
_output_shapes
:	Р

X
out/zeros_1Const*
valueB
*    *
dtype0*
_output_shapes
:

q
out/b
VariableV2*
shape:
*
dtype0*
	container *
shared_name *
_output_shapes
:

Т
out/b/AssignAssignout/bout/zeros_1*
T0*
validate_shape(*
use_locking(*
_class

loc:@out/b*
_output_shapes
:

\

out/b/readIdentityout/b*
T0*
_class

loc:@out/b*
_output_shapes
:

Е

out/MatMulMatMulPlaceholder
out/W/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€

X
out/addAdd
out/MatMul
out/b/read*
T0*'
_output_shapes
:€€€€€€€€€

Q
out/SoftmaxSoftmaxout/add*
T0*'
_output_shapes
:€€€€€€€€€

S
out/out/tagConst*
valueB Bout/out*
dtype0*
_output_shapes
: 
V
out/outHistogramSummaryout/out/tagout/Softmax*
T0*
_output_shapes
: 
p
Placeholder_1Placeholder*
dtype0*
shape:€€€€€€€€€
*'
_output_shapes
:€€€€€€€€€

I
LogLogout/Softmax*
T0*'
_output_shapes
:€€€€€€€€€

P
mulMulPlaceholder_1Log*
T0*'
_output_shapes
:€€€€€€€€€

_
Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
q
SumSummulSum/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
=
NegNegSum*
T0*#
_output_shapes
:€€€€€€€€€
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
V
MeanMeanNegConst*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
]
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
М
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
\
gradients/Mean_grad/ShapeShapeNeg*
T0*
out_type0*
_output_shapes
:
Ш
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:€€€€€€€€€
^
gradients/Mean_grad/Shape_1ShapeNeg*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ц
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
В
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
А
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
И
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
h
gradients/Neg_grad/NegNeggradients/Mean_grad/truediv*
T0*#
_output_shapes
:€€€€€€€€€
[
gradients/Sum_grad/ShapeShapemul*
T0*
out_type0*
_output_shapes
:
Ж
gradients/Sum_grad/SizeConst*
value	B :*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
Я
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
•
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
С
gradients/Sum_grad/Shape_1Const*
valueB:*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
Н
gradients/Sum_grad/range/startConst*
value	B : *
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
Н
gradients/Sum_grad/range/deltaConst*
value	B :*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
ѕ
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
М
gradients/Sum_grad/Fill/valueConst*
value	B :*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
ђ
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
ъ
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
N*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:€€€€€€€€€
Л
gradients/Sum_grad/Maximum/yConst*
value	B :*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
ј
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:€€€€€€€€€
ѓ
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
Р
gradients/Sum_grad/ReshapeReshapegradients/Neg_grad/Neg gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Ь
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*
T0*

Tmultiples0*'
_output_shapes
:€€€€€€€€€

e
gradients/mul_grad/ShapeShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
]
gradients/mul_grad/Shape_1ShapeLog*
T0*
out_type0*
_output_shapes
:
і
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
m
gradients/mul_grad/mulMulgradients/Sum_grad/TileLog*
T0*'
_output_shapes
:€€€€€€€€€

Я
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ч
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

y
gradients/mul_grad/mul_1MulPlaceholder_1gradients/Sum_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€

•
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Э
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
Џ
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*'
_output_shapes
:€€€€€€€€€

а
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€

Ъ
gradients/Log_grad/Reciprocal
Reciprocalout/Softmax.^gradients/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:€€€€€€€€€

Э
gradients/Log_grad/mulMul-gradients/mul_grad/tuple/control_dependency_1gradients/Log_grad/Reciprocal*
T0*'
_output_shapes
:€€€€€€€€€

|
gradients/out/Softmax_grad/mulMulgradients/Log_grad/mulout/Softmax*
T0*'
_output_shapes
:€€€€€€€€€

z
0gradients/out/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
¬
gradients/out/Softmax_grad/SumSumgradients/out/Softmax_grad/mul0gradients/out/Softmax_grad/Sum/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
y
(gradients/out/Softmax_grad/Reshape/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
Ј
"gradients/out/Softmax_grad/ReshapeReshapegradients/out/Softmax_grad/Sum(gradients/out/Softmax_grad/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
У
gradients/out/Softmax_grad/subSubgradients/Log_grad/mul"gradients/out/Softmax_grad/Reshape*
T0*'
_output_shapes
:€€€€€€€€€

Ж
 gradients/out/Softmax_grad/mul_1Mulgradients/out/Softmax_grad/subout/Softmax*
T0*'
_output_shapes
:€€€€€€€€€

f
gradients/out/add_grad/ShapeShape
out/MatMul*
T0*
out_type0*
_output_shapes
:
h
gradients/out/add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
ј
,gradients/out/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/out/add_grad/Shapegradients/out/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
±
gradients/out/add_grad/SumSum gradients/out/Softmax_grad/mul_1,gradients/out/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
£
gradients/out/add_grad/ReshapeReshapegradients/out/add_grad/Sumgradients/out/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

µ
gradients/out/add_grad/Sum_1Sum gradients/out/Softmax_grad/mul_1.gradients/out/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ь
 gradients/out/add_grad/Reshape_1Reshapegradients/out/add_grad/Sum_1gradients/out/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

s
'gradients/out/add_grad/tuple/group_depsNoOp^gradients/out/add_grad/Reshape!^gradients/out/add_grad/Reshape_1
к
/gradients/out/add_grad/tuple/control_dependencyIdentitygradients/out/add_grad/Reshape(^gradients/out/add_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/out/add_grad/Reshape*'
_output_shapes
:€€€€€€€€€

г
1gradients/out/add_grad/tuple/control_dependency_1Identity gradients/out/add_grad/Reshape_1(^gradients/out/add_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/out/add_grad/Reshape_1*
_output_shapes
:

ј
 gradients/out/MatMul_grad/MatMulMatMul/gradients/out/add_grad/tuple/control_dependency
out/W/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:€€€€€€€€€Р
Ї
"gradients/out/MatMul_grad/MatMul_1MatMulPlaceholder/gradients/out/add_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes
:	Р

z
*gradients/out/MatMul_grad/tuple/group_depsNoOp!^gradients/out/MatMul_grad/MatMul#^gradients/out/MatMul_grad/MatMul_1
х
2gradients/out/MatMul_grad/tuple/control_dependencyIdentity gradients/out/MatMul_grad/MatMul+^gradients/out/MatMul_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/out/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€Р
т
4gradients/out/MatMul_grad/tuple/control_dependency_1Identity"gradients/out/MatMul_grad/MatMul_1+^gradients/out/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/out/MatMul_grad/MatMul_1*
_output_shapes
:	Р

b
GradientDescent/learning_rateConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
ь
1GradientDescent/update_out/W/ApplyGradientDescentApplyGradientDescentout/WGradientDescent/learning_rate4gradients/out/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
_class

loc:@out/W*
_output_shapes
:	Р

ф
1GradientDescent/update_out/b/ApplyGradientDescentApplyGradientDescentout/bGradientDescent/learning_rate1gradients/out/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
_class

loc:@out/b*
_output_shapes
:


GradientDescentNoOp2^GradientDescent/update_out/W/ApplyGradientDescent2^GradientDescent/update_out/b/ApplyGradientDescent
*
initNoOp^out/W/Assign^out/b/Assign
L
Merge/MergeSummaryMergeSummaryout/out*
N*
_output_shapes
: "lнеAъ@      .оЅЦ	H)1a†÷AJнБ
До
:
Add
x"T
y"T
z"T"
Ttype:
2	
Г
ApplyGradientDescent
var"TА

alpha"T

delta"T
out"TА" 
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
4
Fill
dims

value"T
output"T"	
Ttype
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
8
FloorMod
x"T
y"T
z"T"
Ttype:	
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	Р
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И*1.5.02
b'unknown'•k
p
PlaceholderPlaceholder*
dtype0*
shape:€€€€€€€€€Р*(
_output_shapes
:€€€€€€€€€Р
`
	out/zerosConst*
valueB	Р
*    *
dtype0*
_output_shapes
:	Р

{
out/W
VariableV2*
shape:	Р
*
dtype0*
	container *
shared_name *
_output_shapes
:	Р

Х
out/W/AssignAssignout/W	out/zeros*
T0*
validate_shape(*
use_locking(*
_class

loc:@out/W*
_output_shapes
:	Р

a

out/W/readIdentityout/W*
T0*
_class

loc:@out/W*
_output_shapes
:	Р

X
out/zeros_1Const*
valueB
*    *
dtype0*
_output_shapes
:

q
out/b
VariableV2*
shape:
*
dtype0*
	container *
shared_name *
_output_shapes
:

Т
out/b/AssignAssignout/bout/zeros_1*
T0*
validate_shape(*
use_locking(*
_class

loc:@out/b*
_output_shapes
:

\

out/b/readIdentityout/b*
T0*
_class

loc:@out/b*
_output_shapes
:

Е

out/MatMulMatMulPlaceholder
out/W/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€

X
out/addAdd
out/MatMul
out/b/read*
T0*'
_output_shapes
:€€€€€€€€€

Q
out/SoftmaxSoftmaxout/add*
T0*'
_output_shapes
:€€€€€€€€€

S
out/out/tagConst*
valueB Bout/out*
dtype0*
_output_shapes
: 
V
out/outHistogramSummaryout/out/tagout/Softmax*
T0*
_output_shapes
: 
p
Placeholder_1Placeholder*
dtype0*
shape:€€€€€€€€€
*'
_output_shapes
:€€€€€€€€€

I
LogLogout/Softmax*
T0*'
_output_shapes
:€€€€€€€€€

P
mulMulPlaceholder_1Log*
T0*'
_output_shapes
:€€€€€€€€€

_
Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
q
SumSummulSum/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
=
NegNegSum*
T0*#
_output_shapes
:€€€€€€€€€
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
V
MeanMeanNegConst*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
]
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
М
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
\
gradients/Mean_grad/ShapeShapeNeg*
T0*
out_type0*
_output_shapes
:
Ш
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:€€€€€€€€€
^
gradients/Mean_grad/Shape_1ShapeNeg*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ц
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
В
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
А
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
И
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
h
gradients/Neg_grad/NegNeggradients/Mean_grad/truediv*
T0*#
_output_shapes
:€€€€€€€€€
[
gradients/Sum_grad/ShapeShapemul*
T0*
out_type0*
_output_shapes
:
Ж
gradients/Sum_grad/SizeConst*
value	B :*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
Я
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
•
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
С
gradients/Sum_grad/Shape_1Const*
valueB:*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
Н
gradients/Sum_grad/range/startConst*
value	B : *
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
Н
gradients/Sum_grad/range/deltaConst*
value	B :*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
ѕ
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
М
gradients/Sum_grad/Fill/valueConst*
value	B :*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
ђ
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
ъ
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
N*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:€€€€€€€€€
Л
gradients/Sum_grad/Maximum/yConst*
value	B :*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
ј
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:€€€€€€€€€
ѓ
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
Р
gradients/Sum_grad/ReshapeReshapegradients/Neg_grad/Neg gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Ь
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*
T0*

Tmultiples0*'
_output_shapes
:€€€€€€€€€

e
gradients/mul_grad/ShapeShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
]
gradients/mul_grad/Shape_1ShapeLog*
T0*
out_type0*
_output_shapes
:
і
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
m
gradients/mul_grad/mulMulgradients/Sum_grad/TileLog*
T0*'
_output_shapes
:€€€€€€€€€

Я
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ч
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

y
gradients/mul_grad/mul_1MulPlaceholder_1gradients/Sum_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€

•
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Э
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
Џ
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*'
_output_shapes
:€€€€€€€€€

а
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€

Ъ
gradients/Log_grad/Reciprocal
Reciprocalout/Softmax.^gradients/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:€€€€€€€€€

Э
gradients/Log_grad/mulMul-gradients/mul_grad/tuple/control_dependency_1gradients/Log_grad/Reciprocal*
T0*'
_output_shapes
:€€€€€€€€€

|
gradients/out/Softmax_grad/mulMulgradients/Log_grad/mulout/Softmax*
T0*'
_output_shapes
:€€€€€€€€€

z
0gradients/out/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
¬
gradients/out/Softmax_grad/SumSumgradients/out/Softmax_grad/mul0gradients/out/Softmax_grad/Sum/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
y
(gradients/out/Softmax_grad/Reshape/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
Ј
"gradients/out/Softmax_grad/ReshapeReshapegradients/out/Softmax_grad/Sum(gradients/out/Softmax_grad/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
У
gradients/out/Softmax_grad/subSubgradients/Log_grad/mul"gradients/out/Softmax_grad/Reshape*
T0*'
_output_shapes
:€€€€€€€€€

Ж
 gradients/out/Softmax_grad/mul_1Mulgradients/out/Softmax_grad/subout/Softmax*
T0*'
_output_shapes
:€€€€€€€€€

f
gradients/out/add_grad/ShapeShape
out/MatMul*
T0*
out_type0*
_output_shapes
:
h
gradients/out/add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
ј
,gradients/out/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/out/add_grad/Shapegradients/out/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
±
gradients/out/add_grad/SumSum gradients/out/Softmax_grad/mul_1,gradients/out/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
£
gradients/out/add_grad/ReshapeReshapegradients/out/add_grad/Sumgradients/out/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

µ
gradients/out/add_grad/Sum_1Sum gradients/out/Softmax_grad/mul_1.gradients/out/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ь
 gradients/out/add_grad/Reshape_1Reshapegradients/out/add_grad/Sum_1gradients/out/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

s
'gradients/out/add_grad/tuple/group_depsNoOp^gradients/out/add_grad/Reshape!^gradients/out/add_grad/Reshape_1
к
/gradients/out/add_grad/tuple/control_dependencyIdentitygradients/out/add_grad/Reshape(^gradients/out/add_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/out/add_grad/Reshape*'
_output_shapes
:€€€€€€€€€

г
1gradients/out/add_grad/tuple/control_dependency_1Identity gradients/out/add_grad/Reshape_1(^gradients/out/add_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/out/add_grad/Reshape_1*
_output_shapes
:

ј
 gradients/out/MatMul_grad/MatMulMatMul/gradients/out/add_grad/tuple/control_dependency
out/W/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:€€€€€€€€€Р
Ї
"gradients/out/MatMul_grad/MatMul_1MatMulPlaceholder/gradients/out/add_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes
:	Р

z
*gradients/out/MatMul_grad/tuple/group_depsNoOp!^gradients/out/MatMul_grad/MatMul#^gradients/out/MatMul_grad/MatMul_1
х
2gradients/out/MatMul_grad/tuple/control_dependencyIdentity gradients/out/MatMul_grad/MatMul+^gradients/out/MatMul_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/out/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€Р
т
4gradients/out/MatMul_grad/tuple/control_dependency_1Identity"gradients/out/MatMul_grad/MatMul_1+^gradients/out/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/out/MatMul_grad/MatMul_1*
_output_shapes
:	Р

b
GradientDescent/learning_rateConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
ь
1GradientDescent/update_out/W/ApplyGradientDescentApplyGradientDescentout/WGradientDescent/learning_rate4gradients/out/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
_class

loc:@out/W*
_output_shapes
:	Р

ф
1GradientDescent/update_out/b/ApplyGradientDescentApplyGradientDescentout/bGradientDescent/learning_rate1gradients/out/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
_class

loc:@out/b*
_output_shapes
:


GradientDescentNoOp2^GradientDescent/update_out/W/ApplyGradientDescent2^GradientDescent/update_out/b/ApplyGradientDescent
*
initNoOp^out/W/Assign^out/b/Assign
L
Merge/MergeSummaryMergeSummaryout/out*
N*
_output_shapes
: ""y
	variableslj
2
out/W:0out/W/Assignout/W/read:02out/zeros:0
4
out/b:0out/b/Assignout/b/read:02out/zeros_1:0"Г
trainable_variableslj
2
out/W:0out/W/Assignout/W/read:02out/zeros:0
4
out/b:0out/b/Assignout/b/read:02out/zeros_1:0"
	summaries

	out/out:0"
train_op

GradientDescentВ'