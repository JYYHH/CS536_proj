from Manager import *

file_paths = [
    # 0
    (
        "/opt/cs536/Blend_Dataset/parse/Concat/FireVid36_NoFireVid7_parse.txt", 
        "/opt/cs536/Blend_Dataset/Concat/FireVid36_NoFireVid7.txt"
    ),

    # 1
    (
        "/opt/cs536/Blend_Dataset/parse/One-by-One/FireVid36_NoFireVid7_parse.txt", 
        "/opt/cs536/Blend_Dataset/One-by-One/FireVid36_NoFireVid7.txt"
    ),
    
    # 2
    (
        "/opt/cs536/Blend_Dataset/parse/One-by-One/FireVid36_NoFireVid7_parse.txt", 
        "/opt/cs536/Blend_Dataset/One-by-One/FireVid36_NoFireVid7.txt"
    ),
]

obj = Globel_Manager()
obj.train_model(
    *file_paths[1],
    20,
    32
)
obj.evaluate_(
    *file_paths[1],
    0.5
)
obj.evaluate_(
    *file_paths[1],
    0.8
)
obj.evaluate_(
    *file_paths[0],
    0.5,
    0
)
obj.evaluate_(
    *file_paths[0],
    0.5,
    1
)
obj.evaluate_(
    *file_paths[2],
    0.5,
    0
)
obj.evaluate_(
    *file_paths[2],
    0.5,
    1
)