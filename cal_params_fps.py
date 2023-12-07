from Model import *
import time
import numpy as np

windows = [1, 2, 5, 10, 25]

for win in windows:
    model = build_ensemble_threeview(inp_len1=win, inp_len2=win, inp_len3=1)
    model.summary()

    num_samples = 50000
    inp1_data = np.random.rand(num_samples, 1, win)
    inp2_data = np.random.rand(num_samples, 1, win)
    inp3_data = np.random.rand(num_samples, 1)

    start_time = time.time()
    predictions = model.predict([inp1_data, inp2_data, inp3_data])
    end_time = time.time()

    inference_time = end_time - start_time
    fps = num_samples / inference_time

    print(f"Inference Time: {inference_time} seconds")
    print(f"FPS: {fps}")