from A import *
from MSM_validation import *

#0 : Two ellipes, 1 : N moons, 2 : 2 ellipes + 2 moons (4D), 
#3 : Iris       , 4 : Seeds  , 5 : Breast_Cancer_Wisconsin
#6 : ILPD       , 7 : Wine   , 8 : Turkish

data_name = np.array(["Two concentric ellipes", "three half moons", "Two ellipses and two half moons(4D)", 
				      "Iris"                  , "Seeds"           , "Breast_Cancer_Wisconsin",
                      "ILPD"                  , "Wine"            , "Turkish"])

N = 80
print("N=",N)

for data_id in range(0,9):
    print("-----------",data_name[data_id],"-----------")					  
    MSM_validation(n_samples=N,data_id=data_id,max_iter=100,noise=0.1,level=0)




