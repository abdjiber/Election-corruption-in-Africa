import numpy as np
import pandas as pd
import keras
from keras import initializers
from sklearn import metrics

from utils import utils
import scoring
import constants


class NeuralNets:
    """Neural networks models class.

    This class help to perfom a deep learning supervised
    and semi-supervised learning.
    """

    def __init__(self, x_train, x_val, y_train, y_val):
        utils.initialize_attributes(self,
                                    x_train,
                                    x_val,
                                    y_train,
                                    y_val
                                    )
        self.preds = []
        self.preds_prob = []

    def supervised_learning(self):
        """Helper function for supervised learning"""

        y = pd.get_dummies(pd.Series(self.y_train).astype('category'))
        y_val = pd.get_dummies(pd.Series(self.y_val).astype('category'))

        callback = keras.callbacks.EarlyStopping(patience=5)
        initializer = initializers.RandomNormal(seed=constants.RANDOM_STATE)

        model = keras.layers.Sequential()
        model.add(
              keras.layers.Dense(
                               units=100,
                               input_shape=(self.x_train.shape[1],),
                               activation='relu',
                               kernel_initializer=initializer
                               )
                  )
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(100,
                                     activation='relu',
                                     kernel_initializer=initializer
                                     )
                  )
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(2,
                                     activation='softmax',
                                     kernel_initializer=initializer
                                     )
                  )

        model.compile(optimizer='SGD',
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )
        model.fit(self.x_train,
                  y,
                  epochs=20,
                  callbacks=[callback],
                  verbose=0,
                  validation_data=(self.x_val, y_val)
                  )
        self.preds_prob = model.predict(self.x_val)[:, 1]
        auc = np.round(metrics.roc_auc_score(self.y_val, self.preds_prob), 2)
        self.preds = np.where(self.preds_prob < 0.5, 0, 1)
        acc, recall, auc = scoring.scores(self.y_val,
                                          self.preds,
                                          self.preds_prob
                                          )
        return self.preds_prob, acc, recall, auc

    def semi_supervised_learning(self, threshold=0.9):
        """Helper function for semi-supervised learning"""
        self.x_train, self.y_train = (utils
                                      .get_semi_supervised_data(self.x_train,
                                                                self.x_val,
                                                                self.y_train,
                                                                self.y_val,
                                                                self.preds_prob,
                                                                threshold
                                                                )
                                      )
        return self.supervised_learning()
