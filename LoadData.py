from A import *
from sklearn.preprocessing import StandardScaler
import gen_synthetic_data as gsd

def LoadData(n_samples=100,data_id=0,test_size=0.2,noise=0):
	"""
		Load Data
		0 : Two ellipes, 1 : N moons, 2 : 2 ellipes + 2 moons (4D), 
		3 : Iris       , 4 : Seeds  , 5 : Wine
		
	"""
	# Load Data 
	if   data_id == -3:
		X_train, t_train = gsd.two_classed_but_elongated_data(training_data=True)
		X_test , t_test  = gsd.two_classed_but_elongated_data(training_data=False)		

	elif   data_id == -2:
		X_train, t_train = gsd.gen_data_for_fully_seperable_data()
		X_test , t_test  = gsd.gen_data_for_fully_seperable_data()
	elif data_id == -1:
		X_train, t_train = gsd.gen_data_for_validation_train()
		X_test , t_test  = gsd.gen_data_for_validation_test()
	elif data_id == 0:
		X_train, t_train = gsd.gen_two_concentric_ellipes(n_samples=n_samples,factor=0.40,noise=noise)
		X_test , t_test  = gsd.gen_two_concentric_ellipes(n_samples=n_samples,factor=0.40,noise=noise)
	elif data_id == 1:
		X_train, t_train = gsd.gen_n_moons(n_samples=n_samples,n_moons=3,noise=noise)
		X_test , t_test  = gsd.gen_n_moons(n_samples=n_samples,n_moons=3,noise=noise)
	elif data_id == 2:
		X_train, t_train = gsd.gen_two_moons_and_two_ellipses(n_samples=n_samples,noise=noise)
		X_test , t_test  = gsd.gen_two_moons_and_two_ellipses(n_samples=n_samples,noise=noise)
	elif data_id == 3:
		df = pd.read_csv('iris.data', header=None)
		df.tail()
		X = df.iloc[:, 0:4].values
		t = df.iloc[:, 4].values
	
	elif data_id == 4:
		df = pd.read_csv('seeds.data', header=None)
		df.tail()
		X = df.iloc[:, 0:7].values
		t = df.iloc[:, 7].values
		t = t-1

	elif data_id == 5:
		df = pd.read_csv('wine.data', header=None)
		df.tail()
		X = df.iloc[:, 1:14].values
		t = df.iloc[:, 0].values
		t = t-1	
	
	elif data_id == 6:
		df = pd.read_csv('sonar.data', header=None)
		df.tail()
		X = df.iloc[:, 0:60].values
		t = df.iloc[:, 60].values

	elif data_id == 7:
		df = pd.read_csv('abalone.data', header=None)
		df.tail()
		X = df.iloc[:, 1:9].values
		t = df.iloc[:, 0].values
	
	elif data_id == 8:
		df = pd.read_csv('breast_cancer_wisconsin.data', header=None)
		df.tail()
		X = df.iloc[:, 1:10].values
		t = df.iloc[:, 10].values
		t = t/2-1

	elif data_id == 9:
		df = pd.read_csv('landmine.data', header=None)
		df.tail()
		X = df.iloc[:, 0:3].values
		t = df.iloc[:, 3].values		
		t = t-1		

	elif data_id == 10:
		df = pd.read_csv('ILPD.data', header=None)
		df.tail()
		X = df.iloc[:, 0:10].values
		t = df.iloc[:, 10].values
		t = t-1
	
	elif data_id == 11:
		df = pd.read_csv('bupa.data', header=None)
		df.tail()
		X = df.iloc[:, 0:6].values
		t = df.iloc[:, 6].values
		t = t-1
	
	elif data_id == 12:
		df = pd.read_csv('cmc.data', header=None)
		df.tail()
		X = df.iloc[:, 0:9].values
		t = df.iloc[:, 9].values
		t = t-1
	
	elif data_id == 13:
		df = pd.read_csv('ionosphere.data', header=None)
		df.tail()
		X = df.iloc[:, 0:34].values
		t = df.iloc[:, 34].values		
	
	elif data_id == 14:
		df = pd.read_csv('turkish.data', header=None)
		df.tail()
		X = df.iloc[:, 1:50].values
		t = df.iloc[:, 0].values		

	# Separate training and test data
	if data_id >= 3:
		X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_size, stratify=t)
		if data_id >= 4:
			scaler = StandardScaler()
			X_train = scaler.fit_transform(X_train)
			X_test  = scaler.transform(X_test)
	
	return X_train, X_test, t_train, t_test