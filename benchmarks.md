# Benchmarks

```
========== Same Value Tensor Initialization ==========

Object Flat						        : 0.0029510650015436113 seconds
Table Nested					        : 0.001206543000880629 seconds
Table Nested Efficient			  : 0.001230360995978117 seconds
Table Nested Efficient IPairs	: 0.0012107809982262552 seconds
Table Mixed						        : 0.0012105519929900765 seconds

========== Identity Tensor Initialization ==========

Object Flat						        : 0.003273945986293256 seconds
Table Nested					        : 0.0012257050001062452 seconds
Table Nested Efficient			  : 0.0010426390008069575 seconds
Table Nested Efficient IPairs	: 0.0009710309864021838 seconds
Table Mixed						        : 0.0012561380094848573 seconds

========== Random Normal Tensor Initialization - Dimension Size From Smallest to Largest ==========

Object Flat						        : 0.010358948009088636 seconds
Table Nested					        : 0.008691198006272317 seconds
Table Nested Efficient			  : 0.008327862990554422 seconds
Table Nested Efficient IPairs	: 0.007525188000872731 seconds
Table Mixed						        : 0.008394609987735748 seconds

========== Random Normal Tensor Initialization - Dimension Size From Largest to Smallest ==========

Object Flat						        : 0.010345545974560081 seconds
Table Nested					        : 0.00964630200061947 seconds
Table Nested Efficient			  : 0.009423127002082764 seconds
Table Nested Efficient IPairs	: 0.009340057997033 seconds
Table Mixed						        : 0.010387461998034269 seconds

========== Addition With Same Sized Tensor ==========

Object Flat						        : 0.002624087997246534 seconds
Table Nested					        : 0.0040913850022479895 seconds
Table Nested Efficient			  : 0.004003426004201174 seconds
Table Nested Efficient IPairs	: 0.004346565008163452 seconds
Table Mixed						        : 0.14040492300642654 seconds
  
========== Addition With Scalar ==========

Object Flat						        : 0.0021173800132237375 seconds
Table Nested					        : 0.0035990140261128543 seconds
Table Nested Efficient			  : 0.003830275989603251 seconds
Table Nested Efficient IPairs	: 0.00401610501576215 seconds
Table Mixed						        : 0.0967610319936648 seconds

========== Full Sum ==========

Object Flat						        : 0.00029061697889119385 seconds
Table Nested					        : 0.0008137540007010102 seconds
Table Nested Efficient			  : 0.00077779600629583 seconds
Table Nested Efficient IPairs	: 0.0007261719927191734 seconds
Table Mixed						        : 0.05189772801008075 seconds

========== Dimension Sum ==========

Object Flat						        : 0.035122452995274216 seconds
Table Nested					        : 0.09353672800585627 seconds
Table Nested Efficient			  : 0.042314096009358766 seconds
Table Nested Efficient IPairs	: 0.04525559599045664 seconds
Table Mixed						        : 0.14654548100195824 seconds

========== Transpose ==========

Object Flat						        : 0.030782521015498786 seconds
Table Nested					        : 0.05762634299695492 seconds
Table Nested Efficient			  : 0.02704005199484527 seconds
Table Nested Efficient IPairs	: 0.030055498988367617 seconds
Table Mixed						        : 0.10146230699727311 seconds

========== Set Value By Function ==========

Object Flat						        : 0.00001592099666595459 seconds
Table Nested					        : 0.00002218098845332861 seconds
Table Nested Efficient			  : 0.00001635600347071886 seconds
Table Nested Efficient IPairs	: 0.00001694199861958623 seconds
Table Mixed						        : 0.000023511997424066066 seconds

========== Set Value By Index (Not Applicable To Object Flat) ==========

Object Flat						        : 0.000017538995016366244 seconds
Table Nested					        : 0.0000031130132265388968 seconds
Table Nested Efficient			  : 0.000002935004886239767 seconds
Table Nested Efficient IPairs	: 0.000003062000032514334 seconds
Table Mixed						        : 0.0000034019979648292063 seconds

========== Get Value By Function ==========

Object Flat						        : 0.000015686014667153358 seconds
Table Nested					        : 0.000021171004045754672 seconds
Table Nested Efficient			  : 0.000016851010732352732 seconds
Table Nested Efficient IPairs	: 0.00001708101248368621 seconds
Table Mixed						        : 0.000022022987250238656 seconds

========== Get Value By Index (Not Applicable To Object Flat) ==========

Object Flat						        : 0.000016976019833236932 seconds
Table Nested					        : 0.000003255009651184082 seconds
Table Nested Efficient			  : 0.0000030629942193627355 seconds
Table Nested Efficient IPairs	: 0.0000033149891532957555 seconds
Table Mixed						        : 0.0000035509839653968812 seconds

========== Dot Product With Transposed Same Sized Tensor ==========

Object Flat						        : 0.000125386998988688 seconds
Table Nested					        : 0.000028647989965975283 seconds
Table Nested Efficient			  : 0.000029225999023765327 seconds
Table Nested Efficient IPairs	: 0.00002812501508742571 seconds
Table Mixed						        : 0.0001687129936181009 seconds

========== Tensor Dimension Size Expansion ==========

Object Flat						        : 0.02009569002315402 seconds
Table Nested					        : 0.0007851169956848025 seconds
Table Nested Efficient			  : 0.0008162499871104956 seconds
Table Nested Efficient IPairs	: 0.0008028960041701794 seconds
Table Mixed						        : 0.07636554500786588 seconds

========== Tensor Number Of Dimension Expansion ==========

Object Flat						        : 0.029197904011234642 seconds
Table Nested					        : 0.008555947993882 seconds
Table Nested Efficient			  : 0.008506082000676542 seconds
Table Nested Efficient IPairs	: 0.008343759002164006 seconds
Table Mixed						        : 0.13834616999141872 seconds

========== Permutation ==========

Object Flat						        : 0.0032147329859435556 seconds
Table Nested					        : 0.00578366999514401 seconds
Table Nested Efficient			  : 0.003186068998184055 seconds
Table Nested Efficient IPairs	: 0.0031632720027118923 seconds
Table Mixed						        : 0.004666299985256046 seconds
```
