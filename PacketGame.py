from Manager import *

obj = Globel_Manager()
obj.train_model(
    "/opt/cs536/Blend_Dataset/parse/One-by-One/FireVid26_NoFireVid2_parse.txt", 
    "/opt/cs536/Blend_Dataset/One-by-One/FireVid26_NoFireVid2.txt",
    10,
    32,
    True
)
