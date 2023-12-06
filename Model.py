import tensorflow as tf
from tensorflow.keras import layers
from math import sqrt
import numpy as np

def build_ensemble_threeview(inp_len1=5, inp_len2=5, inp_len3=1, conv_units=[32, 32], dense_unit=128):
    """
    build three-view neural network 

    Args:
        inp_len1, inp_len2, inp_len3 (int): input length of three views, 5 for inp1 & inp2 and 1 for inp3 by default
        conv_units (list of int): number of conv1d units
            by default, two conv1d layers with 32 units
        dense_unit (int): number of dense units, 128 by default
    Return:
        tf.keras.Model instance
    """
    inp1 = layers.Input(shape=(None, inp_len1), name="View1-Indepdendent")
    inp2 = layers.Input(shape=(None, inp_len2), name="View2-Predicted")
    inp3 = layers.Input(shape=(None, inp_len3), name="View3-Temporal")
    x1 = inp1
    x2 = inp2 
    x3 = inp3
    for u in conv_units:
        x1 = layers.Conv1D(u, 1, activation="relu")(x1)
        x2 = layers.Conv1D(u, 1, activation="relu")(x2)
    x1 = layers.GlobalMaxPooling1D()(x1)
    x2 = layers.GlobalMaxPooling1D()(x2)
    x = layers.Concatenate()([x1, x2])
    out = layers.Dense(1, activation="sigmoid")(x)
    x4 = layers.Concatenate()([out, x3])
    x4 = layers.Dense(dense_unit, activation="relu")(x4)
    out2 = layers.Dense(1, activation='sigmoid')(x4)
    return tf.keras.Model(inputs=[inp1, inp2, inp3], outputs=out2)


class Temporal_Estimator:
    """
    We assume offline condition, and only one stream to be handled
    And this Estimator has a cold start problem

    win_s: window size of this Estimator
    T: How many frames / rounds in total
    Lambda: trade-off between exploration and exploit
    self.feed_backs: The Redundancy Feedback (vector) given by the Inference Outputs
    self.now_frame: How many rounds handled
    self.sum_r/self.sum_s: sum of redundancy value and selected (times)
    """
    def __init__(self, win_s, T, Lambda = 1):
        self.window_size = win_s
        self.Total_Frame = T
        self.lamb = Lambda
        self.feed_backs = [] # ground truth
        self.decisions = [] # decisions made by our model

        self.now_frame = 0
        self.sum_r = 0
        self.sum_s = 0

    def update(self, FeedBack, Decision): 
        # FeedBack = 1 -> This frame is useful
        # Decision = 1 -> This frame is selected by our model
        #
        self.now_frame += 1
        self.feed_backs.append(FeedBack)
        self.decisions.append(Decision)
        self.sum_r += FeedBack
        self.sum_s += Decision
        if self.now_frame > self.window_size:
            self.sum_r -= self.feed_backs[-(self.window_size + 1)]
            self.sum_s -= self.decisions[-(self.window_size + 1)]

    def get_miu(self, gate_ = 0):
        # Adapt for the cold start problem
        # 
        if gate_ == 0:
            if self.now_frame >= self.window_size:
                return (
                    self.sum_r / self.window_size 
                    + 
                    self.lamb * sqrt( # lambda for trade-off
                        3 
                        * 
                        self.Total_Frame 
                        / 
                        (
                            2 * self.sum_s 
                            + 
                            1 # avoid zero division
                        )
                    )
                )
            else:
                return (
                    self.sum_r / (self.now_frame + 1) # change
                    + 
                    self.lamb * sqrt(
                        3 
                        * 
                        self.Total_Frame 
                        / 
                        (
                            2 * self.sum_s 
                            * 
                            self.window_size / (self.now_frame + 1) # ajustment
                            + 
                            1 # avoid zero division
                        )
                    )
                )
        else:
            # throwing away the dependency
            if self.now_frame >= self.window_size:
                return (
                    self.sum_r / self.window_size 
                )
            else:
                return (
                    self.sum_r / (self.now_frame + 1) # change
                )

class Contextual_Predictor:
    """
    We assume offline condition, and only one stream to be handled

    win_s: window size to use for the parser's output
    T: How many frames / rounds in total
    self.model: the tensorflow model for the Predictor
    
    """
    def __init__(self, win_s):
        self.model = build_ensemble_threeview(
                inp_len1 = win_s, 
                inp_len2 = win_s, 
                inp_len3 = 1, # This is the output of Temporal_Estimator
                conv_units = [32, 32], 
                dense_unit = 128
            )
        # compile the model, hyperparameters are default ones
        self.model.compile(
            optimizer = 'adam',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )
        self.window_size = win_s

    def save_to(self, path):
        self.model.save(path)

    def load_from(self, path):
        self.model = keras.models.load_model(path)
    
    def train_(self, I_meta, P_meta, mius, labels, epochs = 10, batch_size = 32):
        """
        I_meta: (T, win_s)
        P_meta: (T, win_s)
        mius: (T, 1)
        labels (T, 1)
        """
        self.model.fit(
            np.array([
                np.array(I_meta).reshape(-1, self.window_size),
                np.array(P_meta).reshape(-1, self.window_size),
                np.array(mius).reshape(-1, 1)
            ]), # X
            np.array(labels).reshape(-1, 1), # Y
            epochs = epochs,
            batch_size = batch_size
        )

    def infer_(self, I_window, P_window, miu):
        """
        I_window: (win_s)
        P_window: (win_s)
        miu: (1)

        return: (1)
        """
        return self.model.predict(
            np.array([
                np.array(I_window), 
                np.array(P_window), 
                np.array(miu)
            ]), # X, but batch_size = 1
        )



