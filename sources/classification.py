import pandas as pd
import time

from machine_learning import Model
from utils import utils


class Classification:

    def __init__(self, list_models, x_train, x_val, y_train, y_val):
        utils.initialize_attributes(self, x_train, x_val, y_train, y_val)
        self.list_models = list_models
        self.list_Models = {}

    def get_vars(self):
        preds = {}
        df_scores = pd.DataFrame(columns=['Accuracy', 'Recall', 'AUC',
                                          'Execution time (s)'],
                                 index=list(self.list_models.keys()))
        recall = '_'
        acc = '_'
        return preds, df_scores, recall, acc

    def run_supervised_learning(self):
        preds, df_scores, recall, acc = self.get_vars()
        for _, (model_name, model) in enumerate(self.list_models.items()):
            print('\n' + model_name)
            if model_name != 'Neural Nets':
                model_ = Model(model, self.x_train, self.x_val,
                               self.y_train, self.y_val)
            else:
                model_ = model
            time_start = time.time()
            preds_prob, acc, recall, auc = model_.supervised_learning()
            time_end = time.time()

            execution_time = time_end - time_start
            df_scores.loc[model_name, :] = [acc, recall, auc, execution_time]
            preds[model_name] = preds_prob
            self.list_Models[model_name] = model_
        return df_scores, preds

    def run_semi_supervised_learning(self, threshold=0.9):
        preds, df_scores, recall, acc = self.get_vars()
        for model_name, model_ in self.list_Models.items():
            print('\n' + model_name)
            time_start = time.time()
            preds_prob, acc, recall, auc = (model_
                                            .semi_supervised_learning(threshold)
                                            )
            time_end = time.time()
            execution_time = time_end - time_start
            df_scores.loc[model_name, :] = [acc, recall, auc, execution_time]
            preds[model_name] = preds_prob
        return df_scores, preds
