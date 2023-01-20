import pandas as pd
import numpy as np
import xgboost
import sklearn

class Pipeline:
    def __init__(self):
        self.final_test = pd.read_csv('Data/test.csv', index_col='PassengerId')
        self.train = pd.read_csv('Data/train.csv', index_col='PassengerId')
        self.train_test_split_ratio= 0.2
        

        for col in self.train.columns:
            if self.train[col].dtype == 'object':
                self.train[col] = self.train[col].astype('category')
                self.train[col] = self.train[col].cat.codes
        for col in self.test.columns:
            if self.test[col].dtype == 'object':
                self.test[col] = self.test[col].astype('category')
                self.test[col] = self.test[col].cat.codes
                
        
        
        
        self.test_x, self.train_x, self.train_y, self.test_y = np.split(self.train.sample(frac=1), [int(self.train_test_split_ratio * len(self.train)), len(self.train)])
        self.test_x = self.test
        self.train_x = self.train.drop('Survived', axis=1)
        self.train_y = self.train['Survived']
        

        
    def run(self):
        # train data
        xgb = xgboost.XGBClassifier()
        xgb.fit(self.train_x, self.train_y)
        predictions = xgb.predict(self.test_x)
        accuracy = sklearn.metrics.accuracy_score(self.test_y, predictions)
        precision = sklearn.metrics.precision_score(self.test_y, predictions)
        
        final_predictions = xgb.predict(self.test_x)
        data = {'PassengerId': self.test.index, 'Survived': predictions}
        data_to_submit = pd.DataFrame(data)
        data_to_submit.to_csv('Data/submission.csv', index=False)
        
        
if __name__ == '__main__':
    Pipeline().run()
    