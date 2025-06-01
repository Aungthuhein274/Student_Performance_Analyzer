from sklearn import linear_model
from sklearn.model_selection import train_test_split
import joblib as jl
from data import student_data

data = student_data[['studytime', 'failures', 'absences', 'Dalc',
                     'Walc', 'goout', 'sex', 'age', 'G1', 'G2', 'G3']]  # dataに学習用データの読み込み
# print(data)
# student_data_isnull.sum()

predict = data['G3']  # 予測したい値をpredictに格納
X = data.drop(columns=[predict.name])  # G3を除いたデータをXに格納
y = predict  # G3をyに格納

# Linerar regession is used because X and y are continuous variables.
# best = 0
# iteration = 0
# while best < 1.0:
#     iteration += 1
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.1)  # データの分割
#     model = linear_model.LinearRegression()
#     model.fit(X_train, y_train)
#
#     score = model.score(X_test, y_test)
#     if score > best:
#         best = score
#         jl.dump(model, 'saved_models/best_model.jl')
#     print(f'Iteration {iteration}, Score: {score:.4f}, Best Score: {best:.4f}')
#
#     if best >= 0.95:
#         break

model = jl.load('saved_models/best_model.jl')  # 保存したモデルの読み込み
print('Model loaded successfully.')
