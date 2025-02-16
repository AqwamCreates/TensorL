```
========== Same Value Tensor Initialization ==========

Object Flat						: 0.0007300040003610775 seconds
Table Nested					: 0.00031923299829941245 seconds
Table Nested Efficient V2		: 0.0003277989983325824 seconds
Table Nested Efficient V3		: 0.0001726919983047992 seconds
Table Nested Efficient IPairs V2: 0.00031967599934432653 seconds
Table Mixed						: 0.00031875200220383705 seconds

========== Identity Tensor Initialization ==========

Object Flat						: 0.000823261997429654 seconds
Table Nested					: 0.00031959600048139694 seconds
Table Nested Efficient V2		: 0.00030446999997366217 seconds
Table Nested Efficient V3		: 0.00018443199980538338 seconds
Table Nested Efficient IPairs V2: 0.0003205989964772016 seconds
Table Mixed						: 0.00033451799245085565 seconds

========== Random Normal Tensor Initialization - Dimension Size From Smallest to Largest ==========

Object Flat						: 0.001877543002483435 seconds
Table Nested					: 0.001498212000587955 seconds
Table Nested Efficient V2		: 0.0014760129962814973 seconds
Table Nested Efficient V3		: 0.0015662100043846293 seconds
Table Nested Efficient IPairs V2: 0.0014857530017616228 seconds
Table Mixed						: 0.0015316889982204885 seconds

========== Random Normal Tensor Initialization - Dimension Size From Largest to Smallest ==========

Object Flat						: 0.0019569829996908083 seconds
Table Nested					: 0.0018238040001597256 seconds
Table Nested Efficient V2		: 0.001825392001774162 seconds
Table Nested Efficient V3		: 0.0020421830005943773 seconds
Table Nested Efficient IPairs V2: 0.0018396839965134858 seconds
Table Mixed						: 0.0018453860026784242 seconds

========== Addition With Same Sized Tensor ==========

Object Flat						: 0.0006140400032745674 seconds
Table Nested					: 0.0009720319980988279 seconds
Table Nested Efficient V2		: 0.0009642749989870936 seconds
Table Nested Efficient V3		: 0.014397518000914716 seconds
Table Nested Efficient IPairs V2: 0.0009755119984038174 seconds
Table Mixed						: 0.03754498800262809 seconds

========== Addition With Scalar ==========

Object Flat						: 0.0005717230029404164 seconds
Table Nested					: 0.0008995510020758956 seconds
Table Nested Efficient V2		: 0.0009634379961062223 seconds
Table Nested Efficient V3		: 0.010346639001509174 seconds
Table Nested Efficient IPairs V2: 0.0009547939943149686 seconds
Table Mixed						: 0.026498909002402798 seconds

========== Full Sum ==========

Object Flat						: 0.00008812000160105527 seconds
Table Nested					: 0.0002135190018452704 seconds
Table Nested Efficient V2		: 0.0002027849986916408 seconds
Table Nested Efficient V3		: 0.00019603199965786188 seconds
Table Nested Efficient IPairs V2: 0.00019376899814233184 seconds
Table Mixed						: 0.01318658999807667 seconds

========== Dimension Sum ==========

Object Flat						: 0.007394106999272481 seconds
Table Nested					: 0.025281976996921002 seconds
Table Nested Efficient V2		: 0.008850317997857928 seconds
Table Nested Efficient V3		: 0.008923121001571417 seconds
Table Nested Efficient IPairs V2: 0.008855921003269032 seconds
Table Mixed						: 0.03704834999574814 seconds

========== Transpose ==========

Object Flat						: 0.007173846000805497 seconds
Table Nested					: 0.014665363997337408 seconds
Table Nested Efficient V2		: 0.0065449230017839 seconds
Table Nested Efficient V3		: 0.0064543009974295275 seconds
Table Nested Efficient IPairs V2: 0.006540375998010859 seconds
Table Mixed						: 0.027212323002750054 seconds

========== Set Value By Function ==========

Object Flat						: 0.000008332998841069638 seconds
Table Nested					: 0.00001094299543183297 seconds
Table Nested Efficient V2		: 0.000007718999986536801 seconds
Table Nested Efficient V3		: 0.000007847000379115343 seconds
Table Nested Efficient IPairs V2: 0.00000786100048571825 seconds
Table Mixed						: 0.000012159000616520644 seconds

========== Set Value By Index (Not Applicable To Object Flat) ==========

Object Flat						: 0.000009215001482516527 seconds
Table Nested					: 0.000002050000475719571 seconds
Table Nested Efficient V2		: 0.000002114002127200365 seconds
Table Nested Efficient V3		: 0.000002030999748967588 seconds
Table Nested Efficient IPairs V2: 0.0000020439981017261745 seconds
Table Mixed						: 0.0000020150002092123033 seconds

========== Get Value By Function ==========

Object Flat						: 0.000008339002379216254 seconds
Table Nested					: 0.000010505997925065459 seconds
Table Nested Efficient V2		: 0.00000819999841041863 seconds
Table Nested Efficient V3		: 0.000007930999854579568 seconds
Table Nested Efficient IPairs V2: 0.000008298998000100256 seconds
Table Mixed						: 0.000011172998929396272 seconds

========== Get Value By Index (Not Applicable To Object Flat) ==========

Object Flat						: 0.000009004005114547909 seconds
Table Nested					: 0.000002039005048573017 seconds
Table Nested Efficient V2		: 0.000002182009629905224 seconds
Table Nested Efficient V3		: 0.0000021089951042085887 seconds
Table Nested Efficient IPairs V2: 0.000002234001294709742 seconds
Table Mixed						: 0.0000020809948910027743 seconds

========== Dot Product With Transposed Same Sized Tensor ==========

Object Flat						: 0.000042288001277484 seconds
Table Nested					: 0.000014479003148153425 seconds
Table Nested Efficient V2		: 0.00001559899828862399 seconds
Table Nested Efficient V3		: 0.00001451200048904866 seconds
Table Nested Efficient IPairs V2: 0.000014279004535637796 seconds
Table Mixed						: 0.000050466000684536996 seconds

========== Tensor Dimension Size Expansion ==========

Object Flat						: 0.004712819001288154 seconds
Table Nested					: 0.00016568100138101728 seconds
Table Nested Efficient V2		: 0.00016946499759797008 seconds
Table Nested Efficient V3		: 0.0001628379983594641 seconds
Table Nested Efficient IPairs V2: 0.00016639299923554063 seconds
Table Mixed						: 0.02070227299758699 seconds

========== Tensor Number Of Dimension Expansion ==========

Object Flat						: 0.006651390001643449 seconds
Table Nested					: 0.0016912169987335802 seconds
Table Nested Efficient V2		: 0.0016796800022711978 seconds
Table Nested Efficient V3		: 0.0016954930050997064 seconds
Table Nested Efficient IPairs V2: 0.0016929749975679441 seconds
Table Mixed						: 0.038985579002182934 seconds

========== Permutation ==========

Object Flat						: 0.000603434998774901 seconds
Table Nested					: 0.0011668320000171662 seconds
Table Nested Efficient V2		: 0.000565564997959882 seconds
Table Nested Efficient V3		: 0.0005509739997796714 seconds
Table Nested Efficient IPairs V2: 0.0005722299986518919 seconds
Table Mixed						: 0.0009392689977539703 seconds

```
