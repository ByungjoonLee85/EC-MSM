from A import *

def MSM_validation(n_samples=100,data_id=0,test_size=0.3,max_iter=100,noise=0,level=0,drawing=False):
    # Classification test with MSM, MSMm(0,1,2), SVM, and kNN
    # Output : Accuracy list, Elapsed Time list
    acc_list  = [[] for i in range(9)]
    time_list = [[] for i in range(9)]
    start = time.time()
    for k in range(max_iter):
        if k%10 == 0: print("iter:",k)
        X_train, X_test, t_train, t_test = LoadData(data_id=data_id,test_size=test_size,noise=noise)

        # 1. MSM
        clf = MSM()
        start = time.time()
        clf.fit(X_train,t_train)
        time_list[0].append(time.time()-start)
        acc_list [0].append(clf.predict(X_test,t_test))
                
        # 2. MSM modified level=0
        clf = MSM_modified()
        start = time.time()
        clf.fit(X_train,t_train)        
        time_list[1].append(time.time()-start)
        acc_list [1].append(clf.predict(X_test,t_test))
        
        # 3. MSM modified level=1
        clf = MSM_modified(level=1)
        start = time.time()
        clf.fit(X_train,t_train)        
        time_list[2].append(time.time()-start)
        acc_list [2].append(clf.predict(X_test,t_test,level=1))
        
        # 4. MSM modified level=2
        clf = MSM_modified(level=2)
        start = time.time()
        clf.fit(X_train,t_train)        
        time_list[3].append(time.time()-start)
        acc_list [3].append(clf.predict(X_test,t_test))
        
        # 3. SVM														
        clf = SVC(C=1.0, cache_size=2000, class_weight=None, coef0=0.0,
		          decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
				  max_iter=-1, probability=False, random_state=1, shrinking=True,
				  tol=0.001, verbose=False)
																
        start = time.time()
        clf.fit(X_train,t_train)
        time_list[4].append(time.time()-start)
        predict_svm = clf.predict(X_test)
        SVM_errors = abs(predict_svm-t_test)
        acc_list[4].append(np.round((1-np.mean(SVM_errors))*100,2))

        # 5. kNN
        neigh = KNeighborsRegressor(n_neighbors=4)
        neigh.fit(X_train,t_train)        
        time_list[5].append(time.time()-start)
        predict_knn = neigh.predict(np.array(X_test))        
        knn_errors = abs(predict_knn-t_test)
        acc_list[5].append(np.round((1-np.mean(knn_errors))*100,2))  

        # RF
        clf = RandomForestRegressor(n_estimators=100      , criterion='squared_error'    , max_depth=None              ,
									min_samples_split=2    , min_samples_leaf=1 , min_weight_fraction_leaf=0.0, 
                                    max_leaf_nodes=None, min_impurity_decrease=0.0   ,
									bootstrap =True    , oob_score=False             ,  
									random_state=None  , verbose= 0                  ,
									warm_start=False)

        start = time.time()
        clf.fit(X_train,t_train)
        time_list[6].append(time.time()-start)
        predict_rf = clf.predict(X_test)
        rf_errors = abs(predict_rf-t_test)
        acc_list[6].append(np.round((1-np.mean(rf_errors))*100,2))	
	
		# DNN
        clf = DNN(input_size =X_train.shape[1],hidden_size=100,
        		  output_size=len((np.unique(t_train)).astype(np.intp)))									
        start = time.time()
        clf.fit(X_train,t_train)
        time_list[7].append(time.time()-start)
        acc_list [7].append(clf.accuracy(X_test,t_test)*100)          

        # TabNet
        clf = TabNetClassifierSK(epochs=200, batch_size=None, verbose=0)
        start = time.time()
        clf.fit(X_train,t_train)
        time_list[8].append(time.time()-start)
        acc_list [8].append(accuracy_score(t_test, clf.predict(X_test))*100)

    print("MSM         : acc = ", format(np.mean(acc_list[0]),".2f"),"std = ", format(np.std(acc_list[0]),".2f"),"elapsed time = ", format(np.mean(time_list[0]),".5f"))
    print("MSMm(level0): acc = ", format(np.mean(acc_list[1]),".2f"),"std = ", format(np.std(acc_list[1]),".2f"),"elapsed time = ", format(np.mean(time_list[1]),".5f"))
    print("MSMm(level1): acc = ", format(np.mean(acc_list[2]),".2f"),"std = ", format(np.std(acc_list[2]),".2f"),"elapsed time = ", format(np.mean(time_list[2]),".5f"))
    print("MSMm(level2): acc = ", format(np.mean(acc_list[3]),".2f"),"std = ", format(np.std(acc_list[3]),".2f"),"elapsed time = ", format(np.mean(time_list[3]),".5f"))    
    print("SVM         : acc = ", format(np.mean(acc_list[4]),".2f"),"std = ", format(np.std(acc_list[4]),".2f"),"elapsed time = ", format(np.mean(time_list[4]),".5f"))
    print("kNN         : acc = ", format(np.mean(acc_list[5]),".2f"),"std = ", format(np.std(acc_list[5]),".2f"),"elapsed time = ", format(np.mean(time_list[5]),".5f"))
    print("RF          : acc = ", format(np.mean(acc_list[6]),".2f"),"std = ", format(np.std(acc_list[6]),".2f"),"elapsed time = ", format(np.mean(time_list[6]),".5f"))
    print("DNN         : acc = ", format(np.mean(acc_list[7]),".2f"),"std = ", format(np.std(acc_list[7]),".2f"),"elapsed time = ", format(np.mean(time_list[7]),".5f"))
    print("TabNet      : acc = ", format(np.mean(acc_list[8]),".2f"),"std = ", format(np.std(acc_list[8]),".2f"),"elapsed time = ", format(np.mean(time_list[8]),".5f"))