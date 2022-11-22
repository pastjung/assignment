#PLEASE WRITE THE GITHUB URL BELOW!
# https://github.com/pastjung/assignment.git

import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn import svm

def load_dataset(dataset_path):
	#To-Do: Implement this function
        csv_file = pd.read_csv(dataset_path)
        return csv_file
        
def dataset_stat(dataset_df):	
	#To-Do: Implement this function
        class0=0
        class1=0
        for i in dataset_df.target:
                if i == 0:
                        class0 += 1
                elif i == 1:
                        class1 += 1
                
        return len(dataset_df.columns), class0, class1 


def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
        data = dataset_df.drop(['target'], axis = 1)
        x_train, x_test, y_train, y_test = train_test_split(data, dataset_df["target"], test_size = testset_size)
        return x_train, x_test, y_train, y_test
        

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function

        # 트레이닝
        dt_cls = DecisionTreeClassifier()
        dt_cls.fit(x_train, y_train)

        # 분석
        accuracy = accuracy_score(y_test, dt_cls.predict(x_test))
        precision = precision_score(y_test, dt_cls.predict(x_test))
        recall = recall_score(y_test, dt_cls.predict(x_test))

        return accuracy, precision, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function

        # 트레이닝
        rf_cls = RandomForestClassifier()
        rf_cls.fit(x_train, y_train)

        # 분석
        accuracy = accuracy_score(y_test, rf_cls.predict(x_test))
        precision = precision_score(y_test, rf_cls.predict(x_test))
        recall = recall_score(y_test, rf_cls.predict(x_test))

        return accuracy, precision, recall

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function

        # 트레이닝
        svm_pipe = make_pipeline(
                StandardScaler(), svm.SVC()
                )
        svm_pipe.fit(x_train, y_train)

        # 분석
        accuracy = accuracy_score(y_test, svm_pipe.predict(x_test))
        precision = precision_score(y_test, svm_pipe.predict(x_test))
        recall = recall_score(y_test, svm_pipe.predict(x_test))
        
        return accuracy, precision, recall  

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)


if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
        
