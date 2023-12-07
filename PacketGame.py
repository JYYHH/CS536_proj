from Manager import *

# file_paths = [
#     # 0
#     (
#         "/opt/cs536/Blend_Dataset/parse/Concat/FireVid36_NoFireVid7_parse.txt", 
#         "/opt/cs536/Blend_Dataset/Concat/FireVid36_NoFireVid7.txt"
#     ),

#     # 1
#     (
#         "/opt/cs536/Blend_Dataset/parse/One-by-One/FireVid36_NoFireVid7_parse.txt", 
#         "/opt/cs536/Blend_Dataset/One-by-One/FireVid36_NoFireVid7.txt"
#     ),
    
#     # 2
#     (
#         "/opt/cs536/Blend_Dataset/parse/One-by-One/FireVid36_NoFireVid7_parse.txt", 
#         "/opt/cs536/Blend_Dataset/One-by-One/FireVid36_NoFireVid7.txt"
#     ),
# ]

file_paths = [
    (   
        "/opt/cs536/Blend_Dataset/merged_Concat_equal_train_parse.txt",
        "/opt/cs536/Blend_Dataset/merged_Concat_equal_train.txt"
    ),
    (
        "/opt/cs536/Blend_Dataset/merged_Concat_equal_test_parse.txt",
        "/opt/cs536/Blend_Dataset/merged_Concat_equal_test.txt"
    ),
    (
        "/opt/cs536/Blend_Dataset/parse/2s_equal/train",
        "/opt/cs536/Blend_Dataset/2s_equal/train"
    ),
    (
        "/opt/cs536/Blend_Dataset/parse/2s_equal/test",
        "/opt/cs536/Blend_Dataset/2s_equal/test"
    )
]
obj = Globel_Manager()
obj.train_model(
    *file_paths[2],
    3,
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
    *file_paths[3],
    0.5
)
obj.evaluate_(
    *file_paths[3],
    0.8
)
# obj.evaluate_(
#     *file_paths[0],
#     0.5,
#     0
# )
# obj.evaluate_(
#     *file_paths[0],
#     0.5,
#     1
# )
# obj.evaluate_(
#     *file_paths[2],
#     0.5,
#     0
# )
# obj.evaluate_(
#     *file_paths[2],
#     0.5,
#     1
# )