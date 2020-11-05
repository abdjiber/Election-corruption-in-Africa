from utils import utils
import scoring


class Model:
    """Class for performing supervised and semi-supervised learning"""

    def __init__(self, model, x_train, x_val, y_train, y_val):
        utils.initialize_attributes(self, x_train,
                                    x_val, y_train, y_val
                                    )
        self.model = model
        self.preds = []
        self.preds_prob = []

    def fit(self, params=None):
        if params is not None:
            self.model.fit(self.x_train, self.y_train, **params)
        else:
            self.model.fit(self.x_train, self.y_train)
        return self

    def predict(self):
        self.preds = self.model.predict(self.x_val)
        self.preds_prob = self.model.predict_proba(self.x_val)[:, 1]

    def fit_predict(self):
        self.fit()
        self.predict()
        return self

    def supervised_learning(self):
        self.fit_predict()
        acc, recall, auc = scoring.scores(self.y_val,
                                          self.preds,
                                          self.preds_prob
                                          )
        return self.preds_prob, acc, recall, auc

    def semi_supervised_learning(self, threshold=0.9):
        self.x_train, self.y_train = utils.get_semi_supervised_data(
                                          self.x_train,
                                          self.x_val,
                                          self.y_train,
                                          self.y_val,
                                          self.preds_prob,
                                          threshold
                                          )
        preds_prob, acc, recall, auc = self.supervised_learning()
        return preds_prob, acc, recall, auc
