Rough CFG for config files: 
| is concat, 
+ is one or more, 
* is zero or more, 
? is one or zero)

^ is OR (i.e. A:B^C is short for A:B, A:C), 

NVPairs occur as name value pairs, all specific name value pairs number in dozen, so skipping

NVPairs : [Name] ":" [Value] [Comment]* [NewLine]

[Name]  : [Char]+ " "
[Value] : [Number|Vec]
[Vec] : [Vec2 | Vec3 ]
[Vec] : [ char Integer "," Integer "," char ]
[Vec3] : [ char Integer "," Integer "," Integer "," char ]
[Number] : [Integer | Double] 

==========================

[Config File ] :  [[Comment]* | [Block Description] | [WhiteSpace + NewLine]*]+

[Comment] : ["#" + Char* + NewLine] (any block can contain [Comment]* implicitly, ignored from now on)
 
[Block Description] : [ [NW Desc] |  [Layer Desc]+ ] ^ [ [Layer Desc]+ | [NW Desc] | [Layer Desc]+ ] ^ [ [Layer Desc]+ | [NW Desc] ]

[NW Desc] : [ "->NetworkDescription" [NVPairs] "->EndNetworkDescription"] [NewLine]*

[Layer Desc] : [Conv Desc] ^ [AvgPool Desc] ^ [MaxPool Desc] ^ [FullyConn Desc]

[Conv Desc] : [PConv Desc] ^ [FConv Desc]

[PConv Desc] : [  [FConv Desc] [Layer Desc]+ [ConnTable] [FConv Desc]]

[FConv Desc] : [  "->ConvLayer" [NVPairs] "->EndConvLayer" ]

[ConnTable] : ["->ConnectionTable" [NVPairs] "->EndConnectionTable"]

[AvgPool Desc] : ["->AveragePoolingLayer" [NVPairs] "->EndAveragePoolingLayer"]

[MaxPool Desc] : ["->MaxPoolingLayer" [NVPairs] "->EndMaxPoolingLayer"]

[FullyConn Desc] : ["->FullyConnectedLayerGroup" [NVPairs] "->EndFullyConnectedLayerGroup"]


==========================
