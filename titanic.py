import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ThreadPoolExecutor
from threading import RLock, Condition
from multiprocessing import Process
import time


class Data:
    def __init__(self, data):
        self.data = data

    def edit(self):
        titanic_data = pd.read_csv(self.data)
        X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
        y = titanic_data.Survived
        X = pd.get_dummies(X)
        X = X.fillna({'Age': X.Age.median()})
        return X, y


class Processes(Process):
    def __init__(self, size=5, ps=0):
        super().__init__()
        self._size = size
        self._queue = []
        self._mutex = RLock()
        self._empty = Condition(self._mutex)
        self._full = Condition(self._mutex)
        self.ps = ps

    @staticmethod
    def fit(X, y, ps):
        t0 = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        clf_rf = RandomForestClassifier()
        params = {'n_estimators': range(20, 31, 2), 'criterion': ['gini', 'entropy'], 'max_depth': range(5, 16, 2)}
        grid_search_cv_clf = GridSearchCV(clf_rf, params, cv=5)
        grid_search_cv_clf.fit(X_train, y_train)
        print('Оптимальные параметры:', grid_search_cv_clf.best_params_)
        best_clf = grid_search_cv_clf.best_estimator_
        print('Точность предсказаний на тестовом множестве:', best_clf.score(X_test, y_test))
        feature_importances = best_clf.feature_importances_
        feature_importances_df = pd.DataFrame({'features': list(X_train), 'feature_importances': feature_importances})
        print('Коэффициенты важности наблюдений о пассажирах:\n',
              feature_importances_df.sort_values('feature_importances', ascending=False))
        print('Время выполнения процесса', ps, ':', time.time() - t0, '\n')

    @staticmethod
    def process_two(X, y, ps):
        with ThreadPoolExecutor(max_workers=3) as pool:
            pool.submit(Processes().fit(X, y, ps))

    def put(self, X, y, ps):
        with self._full:
            while len(self._queue) >= self._size:
                self._full.wait()
            self._queue.append(Processes().fit(X, y, ps))
            self._empty.notify()

    def get(self):
        with self._empty:
            while len(self._queue) == 0:
                self._empty.wait()
            ret = self._queue.pop(0)
            self._full.notify()
            return ret

    @staticmethod
    def process_three(X, y, ps):
        with ThreadPoolExecutor(max_workers=3) as pool:
            pool.submit(Processes().put(X, y, ps))


def main():
    X, y = Data('train.csv').edit()
    p1 = Process(target=Processes().fit, args=(X, y, 1))
    p2 = Process(target=Processes().process_two, args=(X, y, 2))
    p3 = Process(target=Processes().process_three, args=(X, y, 3))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()


if __name__ == '__main__':
    main()
