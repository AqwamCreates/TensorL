```
========== Same Value Tensor Initialization ==========

Object Flat						: 0.0009412359993439168 seconds
Table Nested					: 0.0004853599990019575 seconds
Table Nested Efficient V2		: 0.00048447800101712344 seconds
Table Nested Efficient V3		: 0.0005001880007330328 seconds
Table Nested Efficient IPairs V2: 0.0004896520025795325 seconds
Table Mixed						: 0.0005104450043290854 seconds


  

========== Identity Tensor Initialization ==========

Object Flat						: 0.000932074001757428 seconds
Table Nested					: 0.0004956050013424829 seconds
Table Nested Efficient V2		: 0.0004919029987649992 seconds
Table Nested Efficient V3		: 0.0004929360025562345 seconds
Table Nested Efficient IPairs V2: 0.0005113700008951128 seconds
Table Mixed						: 0.0005161100003169849 seconds


  

========== Random Normal Tensor Initialization - Dimension Size From Smallest to Largest ==========

Object Flat						: 0.0018339639966143296 seconds
Table Nested					: 0.0015600010025082155 seconds
Table Nested Efficient V2		: 0.001558479002560489 seconds
Table Nested Efficient V3		: 0.001551844998029992 seconds
Table Nested Efficient IPairs V2: 0.0015145299973664806 seconds
Table Mixed						: 0.0015287530003115534 seconds


  

========== Random Normal Tensor Initialization - Dimension Size From Largest to Smallest ==========

Object Flat						: 0.0019439730030717329 seconds
Table Nested					: 0.001678370000445284 seconds
Table Nested Efficient V2		: 0.0016688349982723594 seconds
Table Nested Efficient V3		: 0.0016696279984898866 seconds
Table Nested Efficient IPairs V2: 0.0017039850016590209 seconds
Table Mixed						: 0.001729651999194175 seconds


  

========== Addition With Same Sized Tensor ==========

Object Flat						: 0.0008095980004873127 seconds
Table Nested					: 0.0008207590004894882 seconds
Table Nested Efficient V2		: 0.0008876479999162256 seconds
Table Nested Efficient V3		: 0.0008459979982580989 seconds
Table Nested Efficient IPairs V2: 0.0009548350021941587 seconds
Table Mixed						: 0.04015134999994188 seconds


  

========== Addition With Scalar ==========

Object Flat						: 0.0007873489987105132 seconds
Table Nested					: 0.000872719002654776 seconds
Table Nested Efficient V2		: 0.0008979549974901602 seconds
Table Nested Efficient V3		: 0.0009049559995764866 seconds
Table Nested Efficient IPairs V2: 0.0009444179950514808 seconds
Table Mixed						: 0.028736629999475554 seconds


  

========== Full Sum ==========

Object Flat						: 0.00007412000501062721 seconds
Table Nested					: 0.00019734299858100712 seconds
Table Nested Efficient V2		: 0.00019577200117055327 seconds
Table Nested Efficient V3		: 0.00020236099779140205 seconds
Table Nested Efficient IPairs V2: 0.00020394000166561453 seconds
Table Mixed						: 0.01303481100301724 seconds


  

========== Dimension Sum ==========

Object Flat						: 0.006764181002508849 seconds
Table Nested					: 0.026700223000952972 seconds
Table Nested Efficient V2		: 0.008712220997549593 seconds
Table Nested Efficient V3		: 0.008531819999334403 seconds
Table Nested Efficient IPairs V2: 0.008568800999200902 seconds
Table Mixed						: 0.03913595799705945 seconds


  

========== Transpose ==========

Object Flat						: 0.007009361998643726 seconds
Table Nested					: 0.014509541001752951 seconds
Table Nested Efficient V2		: 0.00609644400596153 seconds
Table Nested Efficient V3		: 0.006109193999436684 seconds
Table Nested Efficient IPairs V2: 0.00611298099742271 seconds
Table Mixed						: 0.027180114003713243 seconds


  

========== Set Value By Function ==========

Object Flat						: 0.000007034000591374934 seconds
Table Nested					: 0.000008941998821683228 seconds
Table Nested Efficient V2		: 0.000006638001650571823 seconds
Table Nested Efficient V3		: 0.000006402998114936054 seconds
Table Nested Efficient IPairs V2: 0.0000056590052554383875 seconds
Table Mixed						: 0.000008273001294583081 seconds


  

========== Set Value By Index (Not Applicable To Object Flat) ==========

Object Flat						: 0.000007421999471262097 seconds
Table Nested					: 0.0000016250013140961529 seconds
Table Nested Efficient V2		: 0.0000016330007929354907 seconds
Table Nested Efficient V3		: 0.0000017849943833425642 seconds
Table Nested Efficient IPairs V2: 0.0000016069988487288355 seconds
Table Mixed						: 0.0000016759964637458325 seconds


  

========== Get Value By Function ==========

Object Flat						: 0.0000074010016396641734 seconds
Table Nested					: 0.000008659997256472707 seconds
Table Nested Efficient V2		: 0.000006457999697886407 seconds
Table Nested Efficient V3		: 0.0000065880018519237634 seconds
Table Nested Efficient IPairs V2: 0.0000064169970573857426 seconds
Table Mixed						: 0.000008955997764132917 seconds


  

========== Get Value By Index (Not Applicable To Object Flat) ==========

Object Flat						: 0.000007181999390013516 seconds
Table Nested					: 0.0000018009991617873311 seconds
Table Nested Efficient V2		: 0.000001691001234576106 seconds
Table Nested Efficient V3		: 0.0000017499958630651236 seconds
Table Nested Efficient IPairs V2: 0.000001593001070432365 seconds
Table Mixed						: 0.0000017519976245239377 seconds


  

========== Dot Product With Transposed Same Sized Tensor ==========

Object Flat						: 0.00003522999759297818 seconds
Table Nested					: 0.000011314001167193055 seconds
Table Nested Efficient V2		: 0.000011886002030223608 seconds
Table Nested Efficient V3		: 0.000011031999019905924 seconds
Table Nested Efficient IPairs V2: 0.000011212993995286524 seconds
Table Mixed						: 0.00003987699863500893 seconds


  

========== Tensor Dimension Size Expansion ==========

Object Flat						: 0.004322347000706941 seconds
Table Nested					: 0.0001700460020219907 seconds
Table Nested Efficient V2		: 0.0001697560033062473 seconds
Table Nested Efficient V3		: 0.0001712799962842837 seconds
Table Nested Efficient IPairs V2: 0.00016957199899479746 seconds
Table Mixed						: 0.021021343000466004 seconds


  

========== Tensor Number Of Dimension Expansion ==========

Object Flat						: 0.006415374999050982 seconds
Table Nested					: 0.0015656769968336447 seconds
Table Nested Efficient V2		: 0.001549846000270918 seconds
Table Nested Efficient V3		: 0.0015581789996940642 seconds
Table Nested Efficient IPairs V2: 0.0015433379990281537 seconds
Table Mixed						: 0.039810928999795575 seconds


  

========== Permutation ==========

Object Flat						: 0.0005740830034483224 seconds
Table Nested					: 0.0010898070002440363 seconds
Table Nested Efficient V2		: 0.0005120789969805628 seconds
Table Nested Efficient V3		: 0.0005357959977118299 seconds
Table Nested Efficient IPairs V2: 0.0005227719969116151 seconds
Table Mixed						: 0.0008658949995879084 seconds
```
