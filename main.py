from A import *
from MSM_validation import *

#0 : Two ellipes, 1 : N moons, 2 : 2 ellipes + 2 moons (4D), 
#3 : Iris       , 4 : Seeds  , 5 : Wine
#6 : Sonar      , 7 : Abalone, 8 : Breast_Cancer_Wisconsin
#9 : Landmine   , 10: ILPD   , 11: Liver Disorders
#12: CMC        , 13: Ionosphere, 14: Turkish

data_name = np.array(["Two concentric ellipes", "three half moons", "Two ellipses and two half moons(4D)", 
				      "Iris"                  , "Seeds"           , "Wine"                               ,
                      "Sonar"                 , "Abalone"         , "Breast_Cancer_Wisconsin",
                      "Landmine"              , "ILPD"            , "Liver Disorders",
                      "CMC"                   , "Ionosphere"      , "Turkish"])

N = 80
print("N=",N)
data_id = 1

print("-----------",data_name[data_id],"-----------")					  
MSM_validation(n_samples=N,data_id=data_id,max_iter=30,noise=0.1,level=0)

#print("N=", N)-
##data_id=5
#for data_id in range(0,6):
#    if data_id==-1 or data_id==-2:
#        print("-----------Valdiation data-----------")					  
#    else:
#        print("-----------",data_name[data_id],"-----------")					  
#    MSM_validation(n_samples=N,data_id=data_id,max_iter=10,noise=0.1,level=0)
##MSM_validation(n_samples=N,data_id=data_id,max_iter=10,noise=0.0,level=1)



