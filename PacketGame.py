from Manager import *

obj = Globel_Manager()
obj.train_model(
    "/opt/cs536/Blend_Dataset/parse/Concat/FireVid36_NoFireVid7_parse.txt", 
    "/opt/cs536/Blend_Dataset/Concat/FireVid36_NoFireVid7.txt",
    10,
    32,
    True
)
# obj.evaluate_(
#     "/opt/cs536/Blend_Dataset/parse/Concat/FireVid36_NoFireVid7_parse.txt", 
#     "/opt/cs536/Blend_Dataset/Concat/FireVid36_NoFireVid7.txt",
#     0.5
# )