import pandas as pd
import numpy as np
import xgboost
import sklearn
from sklearn.ensemble import RandomForestClassifier

class Pipeline:
    def __init__(self):
        self.final_test = pd.read_csv('Data/test.csv', index_col='PassengerId')
        self.train = pd.read_csv('Data/train.csv', index_col='PassengerId')
        self.train_test_split_ratio= 0.2
        # self.train = self.train.dropna()
        # self.final_test = self.final_test.dropna()
        # self.train.drop(['Name'], axis=1, inplace=True)
        # self.final_test.drop(['Name'], axis=1, inplace=True)
        for col in self.train.columns:
            if self.train[col].dtype == 'object':
                self.train[col] = self.train[col].astype('category')
                self.train[col] = self.train[col].cat.codes
        for col in self.final_test.columns:
            if self.final_test[col].dtype == 'object':
                self.final_test[col] = self.final_test[col].astype('category')
                self.final_test[col] = self.final_test[col].cat.codes
                
        
        
        
        self.test_x, self.train_x = np.split(self.train.drop('Survived', axis=1), [int(self.train_test_split_ratio * len(self.train))])
        self.test_y, self.train_y = np.split(self.train['Survived'], [int(self.train_test_split_ratio * len(self.train))])
        

        
    def run(self):
        xgb = xgboost.XGBClassifier()
        xgb.fit(self.train_x, self.train_y)
        predictions = xgb.predict(self.test_x)
        accuracy = sklearn.metrics.accuracy_score(self.test_y, predictions)
        precision = sklearn.metrics.precision_score(self.test_y, predictions)
        recall = sklearn.metrics.recall_score(self.test_y, predictions)
        print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')
        final_predictions = xgb.predict(self.final_test)
        data = {'PassengerId': self.final_test.index, 'Survived': final_predictions}
        data_to_submit = pd.DataFrame(data)
        data_to_submit.to_csv('Data/submission.csv', index=False)
        
        
if __name__ == '__main__':
    Pipeline().run()
    