"""
single layer perceptron classifier, just a simple logistic regression
"""

import numpy as np 

class Perceptron:
    def __init__(self,n,lr,seed = None):
        self.w = np.zeros(n,dtype=float)
        self.b = 0.0
        self.lr = lr
        if seed is not None:
            np.random.seed(seed)


    def sigmoid(self,z):
        out = np.empty_like(z,dtype=float)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0/(1.0+np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez/(1.0+ez)
        return out

    
    def logit(self,X):
        return X @ self.w + self.b
    
    def predict_proba(self,X):
        return self.sigmoid(self.logit(X))
    
    def predict_class(self,X,threshold = 0.5):
        p = self.predict_proba(X)
        return (p>threshold).astype(np.int64)
    
    #binary-cross-entropy
    def loss_bce(self,y,p_hat,l2 = 0.0):
        eps = 1e-12
        p= np.clip(p_hat,eps,1-eps)
        ce = -np.mean(y*np.log(p) + (1-y) * np.log(1-p))
        if l2 > 0.0:
            ce = ce + (l2/(2 * y.shape[0])) * np.sum(self.w ** 2)
        return ce
    
    def gradient(self,X,y,l2=0.0):
        #forward
        z = self.logit(X)
        p = self.sigmoid(z)
        e = p-y
        m = X.shape[0]
        #grad
        dw = (X.T @ e) / m 
        db = np.sum(e) / m
        if l2 > 0.0:
            dw += (l2/m) * self.w
        return dw,db

    def update(self,dw,db):
        self.w = self.w - self.lr*dw
        self.b = self.b - self.lr*db
    
    def train(self,X,y,epochs = 200,l2=0.0,shuffle = True):
        m = X.shape[0]
        history = []
        idx = np.arange(m)
        for _ in range(epochs):
            if shuffle:
                np.random.shuffle(idx)
            Xs,ys = X[idx],y[idx]
            dw,db = self.gradient(X,y,l2=l2)
            self.update(dw,db)

            p = self.predict_proba(X)
            history.append(self.loss_bce(y,p,l2=l2))
        
        return np.array(history)

    def standardize(self,X):
        mu = X.mean(axis = 0,keepdims = True)
        sigma = X.std(axis=0, keepdims=True)
        sigma[sigma == 0] = 1.0
        return (X - mu) / sigma, mu, sigma




