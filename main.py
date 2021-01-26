import numpy as np
import pandas as pd
from preprocess import clean_up_pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest


train_path = "/computational-intelligence-course-final-project/train.csv"
test_path = "/computational-intelligence-course-final-project/test.csv"

train = pd.read_csv(train_path, index_col='Unnamed: 0')
test = pd.read_csv(test_path)

x_train = [clean_up_pipeline(text) for text in train.Text]
x_test = [clean_up_pipeline(text) for text in test.Text]

le = LabelEncoder()
Y = le.fit_transform(train.Category)

text_clf_svc = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                     ('tfidf', TfidfTransformer()),
#                      ('select-K-best', SelectKBest(f_classif, k=200000)),
                     ('clf-svm', LinearSVC(C=0.5, multi_class="crammer_singer", max_iter=5000)),
])

text_clf_svc = text_clf_svc.fit(x_train, Y)

predicted = text_clf_svc.predict(test_x)
print('accuracy on test data = ', np.mean(predicted == test_y))

output = pd.DataFrame({'Id': test.Id,
                       'Category': pred_test})
output.to_csv('submission.csv', index=False)