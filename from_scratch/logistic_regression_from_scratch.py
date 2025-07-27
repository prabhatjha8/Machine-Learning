
import numpy as np

def sigmoid(z):
  return (1.0/(1.0 + np.exp(z)))


class LogisticRegression:
  def __init__(self, X, y):
    self.y = y.reshape((-1,1))
    self.n = X.shape[0]
    self.p = X.shape[1]
    self.X = np.array([[1.0]*(self.n)]+ list(X.T)).T
    self.weights = np.random.normal(0,1, self.p+1).reshape((-1, 1))
    self.learning_rate = 0.01
    self.n_iter = 1000

  def forward_pass(self):
    return (sigmoid(-1*np.matmul(self.X, self.weights)))

  def categorical_cross_entropy_loss(self):
    y_pred = self.forward_pass()
    log_odds = np.log(y_pred/(1-y_pred))
    neg_pred_logs = np.log(1-y_pred)
    return -(np.matmul((self.y).T, log_odds)[0] + np.sum(neg_pred_logs))/self.n

  def gradient(self):
    return (-1/self.n)*np.matmul((self.X).T, self.y - self.forward_pass())


  def update_weight(self):
    self.weights = self.weights - self.learning_rate*self.gradient()

  def call(self):
    for i in range(self.n_iter):
      self.update_weight()

  def predict(self, X_test):
    m = X_test.shape[0]
    return (sigmoid(-1*np.matmul(np.array([[1.0]*m]+ list(X_test.T)).T, self.weights)))

