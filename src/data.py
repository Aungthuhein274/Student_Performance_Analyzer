import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import joblib as jl

student_data = pd.read_csv('data/student-mat.csv')  # データの読み込み

# データの区切りが通常とは違うため；を指定して再びデータの読み込み。
student_data = pd.read_csv('data/student-mat.csv', sep=';')
student_data.info()

student_data_isnull = student_data.isnull()


# print(student_data['sex'].value_counts())

student_data['sex'] = student_data['sex'].map({'F': 0, 'M': 1})
# print(student_data['sex'].unique())   # 結果 → [0 1]
# print(student_data['sex'].dtype)      # 結果 → int64（または int32）
