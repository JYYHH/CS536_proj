from Model import *
import matplotlib.pyplot as plt

class Globel_Manager:
    """
    Manage the Temporal_Estimator and Contextual_Predictor (MODEL PART)
        Along with all the offline inputs and feedbacks

        self.T: how many package in total
        self.lamb: the Lambda parameter for Temporal Estimator
        self.window_cp: window size for Contextual Predictor

        self.get_window(): Build sliding windows for model input
        self.prepare_data(): Build input data for the model from the file
        self.evaluate__(): Given predicts and labels (maybe also filter_ratio for Type0), return the acc of the model
        self.train_model(): train model using data from a file pair <origined from a video>
        self.evaluate_(): evaluate model using data from a file pair, and return the (correct_num, total_num) pair
        self.evaluate_iter(): similar to self.evaluate_(), but this time model runs on autoregressive fasion
    """
    def __init__(self, model_save_path = "default_model.h5", win_s_T = 5, win_s_C = 5, Lambda = 1):
        self.model_save_path = model_save_path
        self.window_te = win_s_T
        self.window_cp = win_s_C
        self.lamb = Lambda
        self.contextual_predictor = Contextual_Predictor(win_s_C)

    def get_window(self, List):
        if len(List) >= self.window_cp:
            return List[-self.window_cp:]
        elif len(List) >= 1:
            return [List[0]] * (self.window_cp - len(List)) + List
        else:
            return [15000] * self.window_cp # default pkg sizes
    
    def prepare_data(self, meta_path, label_path):
        T = len(open(meta_path).readlines())
        tmp_temporal = Temporal_Estimator(self.window_te, T, self.lamb)
        meta_file = open(meta_path)
        label_file = open(label_path)
        I_frame_Q, P_frame_Q = [], []
        I_meta, P_meta, mius, labels = [], [], [], []
        
        for i in range(T):
            frame_size, Type = tuple(map(int, meta_file.readline().split(' ')))
            if Type == 1:
                I_frame_Q.append(frame_size)
            else: # All other types are seemed as P frame
                P_frame_Q.append(frame_size)
            label = int(label_file.readline())
            if i > 0:
                mius.append(tmp_temporal.get_miu(1))
            else:
                mius.append(1) # default miu
            tmp_temporal.update(label, label) # we assume that we perfect optimal in train stage
            labels.append(label)
            I_meta.append(self.get_window(I_frame_Q))
            P_meta.append(self.get_window(P_frame_Q))

        return  (I_meta, P_meta, mius, labels)

    def evaluate__(self, predicts, labels, ratio, path_name, Type = 0): # Type = 0 -> filter mode / 1 -> regular mode
        predicts, labels = predicts.reshape(-1), labels.reshape(-1)
        if Type == 0:
            index_ = np.argsort(predicts)
            predicts = np.ones(predicts.shape[0]).astype(int)
            predicts[index_[:int(ratio * predicts.shape[0])]] = 0

            # print(predicts, labels)

            # Only predict == 0 && label == 1 -> incorrect
            incorrect_num = np.sum((predicts ^ 1) & labels)
            acc = 1 - incorrect_num / predicts.shape[0]
            print(f"Model running on Test data {path_name}, has the accuracy {acc}\n    On ratio: {ratio}")
            return predicts.shape[0] - incorrect_num, predicts.shape[0]
        else:
            predicts = predicts >= 0.5
            correct_num = np.sum(predicts == labels)
            acc = correct_num / predicts.shape[0]
            print(f"(Normal Mode) Model running on Test data {path_name}, has the accuracy {acc}")
            return correct_num, predicts.shape[0]

    def train_model(self, meta_path, label_path, epochs = 10, batch_size = 32):
        history = self.contextual_predictor.train_(
                                                *self.prepare_data(
                                                    meta_path, 
                                                    label_path
                                                ), 
                                                epochs, 
                                                batch_size
                                            )
        
        # self.contextual_predictor.save_to(self.model_save_path)


    def evaluate_(self, meta_path, label_path, ratio, Type = 0):
        # self.contextual_predictor.load_from(self.model_save_path)
        I_meta, P_meta, mius, labels = self.prepare_data(
                                            meta_path, 
                                            label_path
                                        )
        predicts = self.contextual_predictor.infer_group(
                                                    I_meta, 
                                                    P_meta, 
                                                    mius
                                                )

        return self.evaluate__(
                    np.array(predicts), 
                    np.array(labels), 
                    ratio,
                    meta_path,
                    Type # And you can specify the evaluate type here
                )

    def evaluate_iter(self, meta_path, label_path, ratio):
        T = len(open(meta_path).readlines())
        tmp_temporal = Temporal_Estimator(self.window_te, T, self.lamb)
        # self.contextual_predictor.load_from(self.model_save_path)
        meta_file = open(meta_path)
        label_file = open(label_path)
        I_frame_Q, P_frame_Q = [], []
        predicts, mius, labels = [], [], []

        for i in range(T):
            frame_size, Type = tuple(map(int, meta_file.readline().split(' ')))
            if Type == 1:
                I_frame_Q.append(frame_size)
            else: # All other types are seemed as P frame
                P_frame_Q.append(frame_size)
            label = int(label_file.readline())
            if i > 0:
                mius.append(tmp_temporal.get_miu(1))
            else:
                mius.append(1) # default miu

            NN_out = self.contextual_predictor.infer_(
                                                    self.get_window(I_frame_Q), 
                                                    self.get_window(P_frame_Q),
                                                    mius[-1]
                                                )
            # NN_out \in [0, 1] 
            tmp_temporal.update(label, NN_out >= 0.5) # autoregression fasion in inference stage
            predicts.append(NN_out)
            labels.append(label)
        
        # print(predicts, labels)

        return self.evaluate__(
                    np.array(predicts), 
                    np.array(labels), 
                    ratio,
                    meta_path
                )



