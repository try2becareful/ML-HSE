from collections import defaultdict

import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

sns.set(style='darkgrid')


class Tree:
    def __init__(self, prediction: float):
        self._prediction = prediction

    def predict(self, x):
        return np.full(x.shape[0], self._prediction)


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.loss_derivative2 = lambda y, z: y ** 2 * self.sigmoid(-y * z) * (1 - self.sigmoid(-y * z))

    def fit_new_base_model(self, x, y, predictions):
        base = BaggingRegressor(
            self.base_model_class(**self.base_model_params),
            max_samples=self.subsample,
            n_estimators=10,
            oob_score=True,
        )
        base.fit(x, -(self.loss_derivative(y, predictions)))

        new_p = base.predict(x)

        best_gamma = self.find_optimal_gamma(y, predictions, new_p, )

        predictions += self.learning_rate * new_p

        self.models.append(base)
        self.gammas.append(best_gamma)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        self.gammas.append(1)
        self.models.append(Tree(y_train.mean()))

        train_predictions = np.full(y_train.shape[0], y_train.mean())
        valid_predictions = np.full(y_valid.shape[0], y_valid.mean())

        self.history['train'].append(self.score(x_train, y_train))
        self.history['valid'].append(self.score(x_valid, y_valid))

        if self.early_stopping_rounds is not None:
            self.validation_loss[0] = self.loss_fn(y_valid, valid_predictions)

        for _ in range(1, self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            self.history['train'].append(self.score(x_train, y_train))
            self.history['valid'].append(self.score(x_valid, y_valid))
            if self.early_stopping_rounds is not None:
                valid_predictions += self.gammas[-1] * self.models[-1].predict(x_valid)
                loss = self.loss_fn(y_valid, valid_predictions)
                if (self.validation_loss < loss).sum() == self.validation_loss.shape[0]:
                    break
        if self.plot:
            plt.plot(range(len(self.models) + 1), [0] + self.history['train'], label='РљР°С‡РµСЃС‚РІРѕ РЅР° train')
            plt.plot(range(len(self.models) + 1), [0] + self.history['valid'], label='РљР°С‡РµСЃС‚РІРѕ РЅР° val')

            plt.title('РљР°С‡РµСЃС‚РІРѕ РѕР±СѓС‡РµРЅРёСЏ')
            plt.xlabel('РќРѕРјРµСЂ РёС‚РµСЂР°С†РёРё')
            plt.ylabel('ROC-AUC')

            plt.legend()
            plt.show()

    def predict_proba(self, x):
        res = self.gammas[0] * self.models[0].predict(x)
        for gamma, model in zip(self.gammas[1:], self.models[1:]):
            res += gamma * model.predict(x)
        result = self.sigmoid(res)
        ans = np.array([1 - result, result]).T
        return ans

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        ans = gammas[np.argmin(losses)]
        return ans

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        pass