from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score
import numpy as np
import pandas as pd

#code inspired by
    # https://www.kaggle.com/lbronchal/sentiment-analysis-with-svm


def fit(train,test):
    np.random.seed(1)

    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    np.random.seed(1)

    vectorizer = TfidfVectorizer(min_df = 5,
                                max_df = 0.8,
                                sublinear_tf = True,
                                use_idf = True)

    pipeline_svm = make_pipeline(vectorizer, 
                                SVC(probability=True, kernel="rbf", 
                                class_weight="balanced"))

    grid_svm = GridSearchCV(pipeline_svm,
                        param_grid = {'svc__C': [0.01, 0.1, 1]}, 
                        cv = kfolds,
                        scoring = "accuracy",
                        verbose=1,   
                        n_jobs=-1) 

    grid_svm.fit(train["text"], train["labels"])
    grid_svm.score(test["text"], test["labels"])
    return grid_svm

def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)        

    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred,average = 'macro')
    prec = precision_score(y, pred,average = 'macro')
    rec = recall_score(y, pred,average = 'macro')
    result = { 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    print(result)

def analysis():
    print("analysis! \n")
    data = pd.read_csv("Data/clean.csv")
    data['text'] = data["headline"] + " " + data["summary"]
    train, test = train_test_split(data, test_size=0.2, random_state=1)
    model = fit(train,test)
    report_results(model,test["text"],test["labels"])



