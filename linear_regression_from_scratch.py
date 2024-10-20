import numpy as np

class LinearRegression:
  def __init__(self, X, y):
    self.y = y
    self.n = X.shape[0]
    self.p = X.shape[1]
    self.X = np.array([[1.0]*(self.n)]+ list(X.T)).T
    self.weights = np.random.normal(0,1, self.p+1).reshape((-1, 1))
    self.learning_rate = 0.001
    self.n_iter = 5000

  def forward_pass(self):
    return (np.matmul(self.X, self.weights))

  def mse_loss(self):
    y_pred = self.forward_pass()
    return np.mean((y_pred-y)**2)

  def gradient(self):
    return (-2/self.n)*np.matmul((self.X).T, self.y - self.forward_pass())

  def update_weight(self):
    self.weights = self.weights - self.learning_rate*self.gradient()

  def call(self):
    for i in range(self.n_iter):
      self.update_weight()

  def predict(self, X_test):
    m = X_test.shape[0]
    return(np.matmul(np.array([[1.0]*m]+ list(X_test.T)).T, self.weights))

