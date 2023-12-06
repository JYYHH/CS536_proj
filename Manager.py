from Model import *
import matplotlib.pyplot as plt

class Globel_Manager:
    """
    Manage the Temporal_Estimator and Contextual_Predictor (MODEL PART)
        Along with all the offline inputs and feedbacks

        self.T: how many package in total
        self.lamb: the Lambda parameter for Temporal Estimator
        self.window_cp: window size for Contextual Predictor
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
    
    def prepare_train_data(self, meta_path, label_path, first_time):
        if not first_time:
            self.contextual_predictor.load_from(self.model_save_path)
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

    def train_model(self, meta_path, label_path, epochs = 10, batch_size = 32, first_time = False):
        history = self.contextual_predictor.train_(
                                                *self.prepare_train_data(
                                                    meta_path, 
                                                    label_path,
                                                    first_time
                                                ), 
                                                epochs, 
                                                batch_size
                                            )
        # whether print the loss function of training
        train_loss = history.history['loss']
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, 'bo-', label='Training loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        self.contextual_predictor.save_to(self.model_save_path)

    def evaluate__(self, predicts, labels, ratio):
        filter_index = int(ratio * predicts.shape[0])
        threshold = np.partition(predicts, filter_index)[filter_index]
        predicts = (predicts >= threshold)
        # Only predict == 0 && label == 1 -> incorrect
        acc = 1 - np.sum((predicts ^ 1) & labels)/predicts.shape[0]
        print(f"Model saved in {self.model_save_path}, has the accuracy {acc}\nOn ratio: {ratio}")

    def evaluate_(self, meta_path, label_path, ratio):
        T = len(open(meta_path).readlines())
        tmp_temporal = Temporal_Estimator(self.window_te, T, self.lamb)
        self.contextual_predictor.load_from(self.model_save_path)
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

        self.evaluate__(
            np.array(predicts), 
            np.array(labels), 
            ratio
        )



