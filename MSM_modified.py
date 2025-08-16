from A import *

class MSM_modified:
    def __init__(self,init_num_clusters=5,level=0,drawing=False):
        self.n=None
        self.d=None
        self.c=None
        self.init_num_clusters=init_num_clusters
        self.K=None
        self.mu=None
        self.sigma=None
        self.w=None
        self.eta=None
        self.tau=None
        self.theta=None
        self.C_thr=None
        self.eps_confidence=None
        self.eps=10**(-3)
        # For Multilayer
        self.knn=None        
        self.multilayered=False
        # For debug : 0=both, 1=nearing_only, 2=multilayer_only
        self.level=level
        self.drawing=drawing
        
    def initialize(self,X,y):
        self.n,self.d=X.shape[0],X.shape[1]
        self.c=len(np.unique(y))
        self.K=(self.init_num_clusters*np.ones(self.c)).astype(np.int64)
        self.mu       =np.empty((self.c,self.init_num_clusters,self.d))
        self.sigma    =np.empty((self.c,self.init_num_clusters,self.d))
        self.w        =np.empty((self.c,self.init_num_clusters,self.d,self.d))
        self.detcov   =np.empty((self.c,self.init_num_clusters))
        self.eta      =np.empty((self.c,self.init_num_clusters))
        self.eps_confidence=np.empty((self.c,self.init_num_clusters))        
        self.tau  =self.d
        self.theta=0.0
        self.C_thr=1
        self.knn  =KNeighborsRegressor(n_neighbors=2)
        
    def fit(self,X,y):
        self.initialize(X,y)
        c=self.c

        Kmeans = []        
        for m in range(0,c):
            Xm=X[np.where(y==m)]            
            # Clustering
            Kmeans.append(self.clustering(Xm,m))            
            Km=self.K[m]            
            for l in range(0,Km):
                Xml=Xm[np.where(Kmeans[m].labels_==l)]    
                self.mu[m,l]=Kmeans[m].cluster_centers_[l]                
                self.sigma[m,l],self.w[m,l],self.detcov[m,l]=self.PCA(Xml,self.mu[m,l])                
                #---------------------Nearing Singular Values------------------------#
                # Compute the ratio between consecutive singular values
                #if self.level!=2:
                #    sigma = self.sigma[m,l]
                #    gamma = -1E10
                #    for j in range(len(sigma)-1):                    
                #        if abs(sigma[j+1])<1e-12: 
                #            if j==0: 
                #                gamma=1
                #                break
                #            else   : break
                #        if abs(sigma[j])<1e-12: 
                #            gamma=1
                #            break
                #        
                #        gamma = max(gamma,sigma[j]/sigma[j+1])
                #    #Adjust sigma
                #    for j in range(len(sigma)-1):
                #        sigma[j+1]=gamma*sigma[j]                
                #    self.sigma[m,l]=sigma        
                #--------------------------------------------------------------------#                                                            
                self.eta[m,l]=self.ComputeScalingfactor(Xml,m,l)                                   

        #-------------------Drawing(For Debug)-------------------#
        if self.drawing==True: 
            fig1, ax=plt.subplots()
            
            cm_interval=np.linspace(0,1,c+1)
            for m in range(0,c):
                Xm=X[np.where(y==m)]            
                Km=self.K[m]
                colors = cm.rainbow(np.linspace(cm_interval[m],cm_interval[m+1],Km))
                for l, c in zip(range(0,Km),colors):
                    Xml=Xm[np.where(Kmeans[m].labels_==l)]  
                    plt.scatter(Xml[:,0],Xml[:,1],color=c) # Data
                    plt.scatter((self.mu[m,l])[0],(self.mu[m,l])[1],marker='x',color=c) # Center
                    indicator=(self.w[m,l][0])[0]*(self.w[m,l][1])[1]-(self.w[m,l][1])[0]*(self.w[m,l][0])[1]
                    if indicator >= 0: indicator=abs(np.dot((self.w[m,l])[0],np.array([1,0])))
                    else :             indicator=abs(np.dot((self.w[m,l])[1],np.array([1,0])))
                    a=np.arccos(indicator)*180.0/np.pi                               
                    ell=Ellipse(xy=self.mu[m,l],width =2.0*(self.sigma[m,l])[0],
                                                height=2.0*(self.sigma[m,l])[1],angle=a,fill=False)
                    ax.add_artist(ell)

            plt.show()

        #-------------------Multi-layer determination-------------------#
        if self.level!=1:
            self.eps_confidence.fill(self.eps)        
            self.ConfidenceScore(X,y)        

            count=0
            U, yU = [], []
            for m in range(0,c):
                Xm=X[np.where(y==m)]            
                Km=self.K[m]
                for l in range(0,Km):
                    lind = np.where(Kmeans[m].labels_==l)                 
                    Xml=Xm[lind] 
                    r   = self.anisotropicDist_Normalizedwrtsmallestone(Xml,self.mu[m,l],self.sigma[m,l],self.w[m,l])
                    phi = np.exp(-r/self.eta[m,l])                
                    Uind = lind[0][phi<=self.eps_confidence[m,l]]                                                
                    for uind in Uind:                     
                        U.append(Xm[uind])
                        yU.append(y[uind])
                                        
                    count=count+np.sum((phi<=self.eps_confidence[m,l]).astype(int))
            
            U, yU=np.array(U), np.array(yU)

            if count > 0 and len(np.unique(yU))>1 :
                self.knn.fit(U,yU)
                self.multilayered=True
                #self.knn.fit(X,y)       

    def clustering(self,X,m):
        # Clustering        
        while True:            
            trigger=True
            Km=self.K[m]                        
            Kmeans=cl.KMeans(n_clusters=Km)
            Kmeans.fit(X)
            t_cm=Kmeans.labels_.astype(np.int64)
            
            min_Cml=1e10            
            for k in np.nditer(np.unique(t_cm)):
                idx=np.where(t_cm==k)                
                min_Cml=min(len(idx[0]),min_Cml)                                
                if min_Cml<=self.tau:
                    trigger=False
                    Km-=1
                    self.K[m]=Km
                    break
                
            if trigger: break
            
        return Kmeans
    
    def ComputeScalingfactor(self,X,m,l):                        
        r=self.anisotropicDist_Normalizedwrtsmallestone(X,self.mu[m,l],self.sigma[m,l],self.w[m,l])                              
        rmax=np.max(r)
        
        return rmax/np.log(1/self.eps)
        
    def PCA(self,X,mu):
        X_shifted=X-mu 
        C=np.matmul(X_shifted.T,X_shifted)/X_shifted.shape[0]                
        detC=np.linalg.det(C)
        eig_vals,eig_vecs=LA.eigh(C)
        eig_vals = np.real_if_close(eig_vals, tol=1000)        
        eig_vals[abs(eig_vals)<=1e-12] = 0
        sigma=np.sqrt(np.sort(eig_vals)[::-1])        
        w=eig_vecs[:,eig_vals.argsort()[::-1]]        
        
        return sigma,w,detC      
    
    def anisotropicDist(self,X,mu,sigma,w):
        p=len(sigma)
        sum=0                        
        
        for j in range(0,p):
            projected_dist=0            
            if sigma[j]!=0 : projected_dist=np.dot(X-mu,sigma[j]*w[:,j])/np.dot(sigma[j]*w[:,j],sigma[j]*w[:,j])            
            sum+=projected_dist**2
            
        return np.sqrt(sum)
    
    def anisotropicDist_EDA(self,X,mu,sigma,w,m,l):
        # Assume that sigma is sorted in descending order
        p=len(sigma)
        sum=0                        
        
        # Explained variance ratio (95%)
        variance_ratio = sigma**2/np.sum(sigma**2)
        p_=p
        for i in range(p):
            if variance_ratio[i] < 0.01:
                p_ = i
                break
        
        # Minimum        
        scale_fac = sigma[p_-1]
        reg_fac   = 0.0#np.mean(sigma)
        
        # Average
        #scale_fac = np.mean(sigma[0:p_])
        
        # Median
        #scale_fac = np.median(sigma[0:p_])
        
        # EDA
        sum = self.detcov[m,l]
        
        for j in range(0,p_):            
            projected_dist=0
            sigma_j=sigma[j]+reg_fac            
            if sigma_j!=reg_fac : projected_dist=np.dot(X-mu,sigma_j/scale_fac*w[:,j])/np.dot(sigma_j/scale_fac*w[:,j],sigma_j/scale_fac*w[:,j])            
            sum+=projected_dist**2

        return np.sqrt(sum)

    def anisotropicDist_Normalizedwrtsmallestone(self,X,mu,sigma,w):
        # Assume that sigma is sorted in descending order
        p=len(sigma)
        sum=0                        
        
        # Explained variance ratio (95%)
        variance_ratio = sigma**2/np.sum(sigma**2)
        p_=p
        for i in range(p):
            if variance_ratio[i] < 0.005:
                p_ = i
                break
        
        # Minimum        
        scale_fac = sigma[p_-1]        
        
        # EDA
        X_shifted=X-mu 
        C=np.matmul(X_shifted.T,X_shifted)/X_shifted.shape[0]      
        
        if C.ndim == 0: sum = np.abs(C)
        else:           sum = np.linalg.det(C)
        
        for j in range(0,p_):            
            projected_dist=0
            sigma_j=sigma[j]
            projected_dist=np.dot(X-mu,sigma_j/scale_fac*w[:,j])/np.dot(sigma_j/scale_fac*w[:,j],sigma_j/scale_fac*w[:,j])            
            sum+=projected_dist**2
    
        return np.sqrt(sum)
    
    def membershipscore(self,X,level=0):        
        phi = np.empty((self.c,self.init_num_clusters))        
        phi.fill(-1.0)
        for m in range(0,self.c):
            for l in range(0,self.K[m]):                
                if (level!=2) :
                    r        = self.anisotropicDist_Normalizedwrtsmallestone(X,self.mu[m,l],self.sigma[m,l],self.w[m,l])                                
                    #r        = self.anisotropicDist_EDA(X,self.mu[m,l],self.sigma[m,l],self.w[m,l],m,l)                                
                else:
                    r        = self.anisotropicDist_Normalizedwrtsmallestone(X,self.mu[m,l],self.sigma[m,l],self.w[m,l])                                
                
                phi[m,l] = np.exp(-r/self.eta[m,l])/self.eps                
                
        return phi
    
    def ConfidenceScore(self,X,y):        
        for m in range(0,self.c):            
            XnotinXm=X[np.where(y!=m)]            
            for l in range(0,self.K[m]):
                r   = self.anisotropicDist_Normalizedwrtsmallestone(XnotinXm,self.mu[m,l],self.sigma[m,l],self.w[m,l])
                phi = np.exp(-r/self.eta[m,l])
                self.eps_confidence[m,l]=max(self.eps_confidence[m,l],np.max(phi))                

    def euclideanDist(self,X):
        euc_d = np.zeros((self.c,self.init_num_clusters))        
        euc_d.fill(-1.0)
        for m in range(0,self.c):
            for l in range(0,self.K[m]):
                euc_d[m,l]   = np.linalg.norm(X-self.mu[m,l])                
                
        return euc_d        
    
    def predict(self,X,y,level=0):
        n_test = len(X)
        count = 0
        
        count_for_knn = 0
        for i in range(n_test):
            Xi=X[i]
            phi   = self.membershipscore(Xi,level)            
            euc_d = self.euclideanDist(Xi)            
            max_idx_phi=0
            if np.min(phi)>self.C_thr:                
                max_idx_phi=np.unravel_index(phi.argmax(),phi.shape)                  
            else:
                phi_blended=(1.-self.theta)*phi+self.theta*(self.eps/euc_d)                
                max_idx_phi=np.unravel_index(phi_blended.argmax(),phi_blended.shape)                                       
                
            m,l = max_idx_phi[0], max_idx_phi[1]            
            if phi[m,l]>self.eps_confidence[m,l]:
                test_label =max_idx_phi[0]               
            else:
                if self.multilayered==False: test_label = max_idx_phi[0]
                else:
                    test_label =self.knn.predict(np.array([Xi]))                    
                    count_for_knn+=1
            
            if test_label==int(y[i]):
                count+=1
        
        return 100.*count/n_test

    def predicted_value(self,X,y,level=0):
        n_test = len(X)
        count = 0
                
        count_for_knn = 0
        predicted = np.zeros_like(y)
        for i in range(n_test):
            Xi=X[i]
            phi   = self.membershipscore(Xi,level)            
            euc_d = self.euclideanDist(Xi)            
            max_idx_phi=0
            if np.min(phi)>self.C_thr:                
                max_idx_phi=np.unravel_index(phi.argmax(),phi.shape)                  
            else:
                phi_blended=(1.-self.theta)*phi+self.theta*(self.eps/euc_d)                
                max_idx_phi=np.unravel_index(phi_blended.argmax(),phi_blended.shape)                                       
                
            m,l = max_idx_phi[0], max_idx_phi[1]            
            if phi[m,l]>self.eps_confidence[m,l]:
                test_label =max_idx_phi[0]               
            else:
                if self.multilayered==False: test_label = max_idx_phi[0]
                else:
                    test_label =self.knn.predict(np.array([Xi]))
                    count_for_knn+=1
            
            predicted[i] = test_label
            
        return predicted
    
    def predict_proba(self,X,y,level=0):
        n_test = len(X)
        count = 0
                
        count_for_knn = 0
        predicted = np.zeros_like(y)
        for i in range(n_test):
            Xi=X[i]
            phi   = self.membershipscore(Xi,level)                                    
            euc_d = self.euclideanDist(Xi)            
            max_idx_phi=0
            phi_max = 0
            if np.min(phi)>self.C_thr:                
                max_idx_phi=np.unravel_index(phi.argmax(),phi.shape)                                  
                phi_max    =phi.max()
            else:
                phi_blended=(1.-self.theta)*phi+self.theta*(self.eps/euc_d)                
                max_idx_phi=np.unravel_index(phi_blended.argmax(),phi_blended.shape)                                       
                phi_max    =phi.max()
                
            m,l = max_idx_phi[0], max_idx_phi[1]            
            if phi[m,l]>self.eps_confidence[m,l]:
                test_label =max_idx_phi[0]
                phi_max    =phi.max()               
            else:
                if self.multilayered==False: 
                    test_label = max_idx_phi[0]
                    phi_max    =phi.max()               
                else:
                    test_label =self.knn.predict(np.array([Xi]))                      
                    phi_max    =phi.max()
                    count_for_knn+=1
            
            predicted[i] = np.log(phi_max)
            
        return predicted