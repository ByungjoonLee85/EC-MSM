from A import *
from layers import *
from collections import OrderedDict

class DNN:
	def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
		# Initialize weights
		self.params = {}
		self.params['W1'] = weight_init_std*np.random.randn(input_size,hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = weight_init_std*np.random.randn(hidden_size,hidden_size)
		self.params['b2'] = np.zeros(hidden_size)
		self.params['W3'] = weight_init_std*np.random.randn(hidden_size,output_size)
		self.params['b3'] = np.zeros(output_size)
		
		# Initialize layers
		self.layers = OrderedDict()
		self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
		self.layers['Relu1']   = Relu()
		self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
		self.layers['Relu2']   = Relu()
		self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
		
		self.lastLayer = SoftmaxWithLoss()
		
	def fit(self,X_train,t_train):
		max_iter = 100000
		train_size = X_train.shape[0]
		batch_size = 20
		learning_rate = 0.1
		
		train_loss_list = []
		err_list = []				
		iter_per_epoch = max(np.ceil(train_size/batch_size), 1)
		
		num_epoch = 0
		for i in range(max_iter):
			batch_mask = np.random.choice(train_size, batch_size)
			x_batch = X_train[batch_mask]
			t_batch = t_train[batch_mask]
			
			if i == 0: train_loss_list.append(self.loss(x_batch,t_batch))
			
			# Compute gradient
			grad = self.gradient(x_batch,t_batch)
			
			# Update
			for key in ('W1','b1','W2','b2','W3','b3'):
				self.params[key] -= learning_rate*grad[key]
			
			if i % iter_per_epoch == 0:
				num_epoch += 1
				loss = self.loss(x_batch,t_batch)
				train_loss_list.append(loss)
				error = np.abs(train_loss_list[num_epoch]-train_loss_list[num_epoch-1])
				if error < .5*10**-5:
					#print("Converges at epoch :", num_epoch)
					break
			
			if i == max_iter : print("Not converges!")
		
	def predict(self,x):
		for layer in self.layers.values():
			x = layer.forward(x)
			
		return x
		
	def loss(self,x,t):
		y = self.predict(x)
		
		return self.lastLayer.forward(y,t)
		
	def accuracy(self,x,t):
		y = self.predict(x)
		y = np.argmax(y,axis=1)
		if t.ndim != 1 : t = np.argmax(t, axis=1)
		
		accuracy = np.sum(y==t)/float(x.shape[0])
		return accuracy
		
	def gradient(self,x,t):
		# forward
		self.loss(x,t)
		
		# backward
		dout = 1
		dout = self.lastLayer.backward(dout)
		
		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)
			
		# Save results
		grads = {}
		grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
		grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
		grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
		
		return grads
		