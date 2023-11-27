import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
 
# 生成数据
np.random.seed(10)
X = np.linspace(0, 1, 10)
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.2, len(X))
X_test = np.linspace(0, 1, 100)
y_test = np.sin(2 * np.pi * X_test) + np.random.normal(0, 0.2, len(X_test))
 
# 不同的模型复杂度
degrees = [1, 3, 10]
X = X[:, np.newaxis]
X_test = X_test[:, np.newaxis]
 
# 绘制结果
plt.figure(figsize=(18, 6))
 
for i in range(len(degrees)):
    
    plt.subplot(1, len(degrees), i+1)
 
    degree = degrees[i]
 
    # 标准多项式拟合
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0))
    model.fit(X, y)
    y_poly_pred = model.predict(X_test)
    plt.plot(X_test,y_test, color='red', label='True function')
    plt.plot(X_test, y_poly_pred, color='blue', label='Polynomial fit (no L2)')
    plt.scatter(X, y, color='navy', s=40, marker='o', label="training points")
 
    # 将多项式拟合添加L2正则化
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.1))
    model.fit(X, y)
    y_poly_pred = model.predict(X_test)
    plt.plot(X_test, y_poly_pred, color='orange', label='Polynomial fit (with L2)')
 
    plt.ylim(-2, 2)
    plt.legend(loc='best')
    plt.title("Degree {}\nTrain Score: {:.3f}, Test Score: {:.3f}".format(
        degree, model.score(X, y), model.score(X_test, y_test)))
 
plt.show()