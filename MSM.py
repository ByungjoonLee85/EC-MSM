from A import *

class MSM:
    def __init__(self,init_num_clusters=5,drawing=False):
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
        self.eps=10**(-3)
        # For debug
        self.drawing=drawing
        
    def initialize(self,X,y):
        self.n,self.d=X.shape[0],X.shape[1]
        self.c=len(np.unique(y))
        self.K=(self.init_num_clusters*np.ones(self.c)).astype(np.int64)
        self.mu   =np.empty((self.c,self.init_num_clusters,self.d))
        self.sigma=np.empty((self.c,self.init_num_clusters,self.d))
        self.w    =np.empty((self.c,self.init_num_clusters,self.d,self.d))
        self.eta  =np.empty((self.c,self.init_num_clusters))
        self.tau  =self.d
        self.theta=0.0
        self.C_thr=1
        
    def fit(self,X,y):
        self.initialize(X,y)
        c=self.c
        
        Kmeans=[]
        for m in range(0,c):
            Xm=X[np.where(y==m)]                        
            # Clustering
            Kmeans.append(self.clustering(Xm,m))            
            Km=self.K[m]            
            for l in range(0,Km):
                Xml=Xm[np.where(Kmeans[m].labels_==l)]                
                self.mu[m,l]=Kmeans[m].cluster_centers_[l]                
                self.sigma[m,l],self.w[m,l]=self.PCA(Xml,self.mu[m,l])                                
                self.eta[m,l]=self.ComputeScalingfactor(Xml,m,l)             

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
                
    def clustering(self,X,m):
        # Clustering        
        while True:            
            trigger=True
            Km=self.K[m]                                    
            Kmeans=cl.KMeans(Km)
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
        r=self.anisotropicDist(X,self.mu[m,l],self.sigma[m,l],self.w[m,l])                              
        rmax=np.max(r)
        
        return 1.0#rmax/np.log(1/self.eps)
        
    def PCA(self,X,mu):
        X_shifted=X-mu 
        C=np.matmul(X_shifted.T,X_shifted)/X_shifted.shape[0]        
        eig_vals,eig_vecs=LA.eig(C)
        eig_vals = np.real_if_close(eig_vals,tol=1000)
        eig_vals[abs(eig_vals)<=1e-16] = 0
        sigma=np.sqrt(np.sort(eig_vals)[::-1])                
        w=eig_vecs[:,eig_vals.argsort()[::-1]]
        
        return sigma,w      
    
    def anisotropicDist(self,X,mu,sigma,w):
        p=len(sigma)
        sum=0                        
                
        sigmatmp = sigma/max(sigma)
        for j in range(0,p):            
            projected_dist=0            
            if sigmatmp[j]!=0 : projected_dist=np.dot(X-mu,sigmatmp[j]*w[:,j])/np.dot(sigmatmp[j]*w[:,j],sigmatmp[j]*w[:,j])
            sum+=projected_dist**2
            
        return np.sqrt(sum)
        
    def membershipscore(self,X):        
        phi = np.empty((self.c,self.init_num_clusters))        
        phi.fill(-1.0)
        for m in range(0,self.c):
            for l in range(0,self.K[m]):
                r        = self.anisotropicDist(X,self.mu[m,l],self.sigma[m,l],self.w[m,l])                                
                phi[m,l] = np.exp(-(r/self.eta[m,l])**2)
                
        return phi
    
    def euclideanDist(self,X):
        euc_d = np.zeros((self.c,self.init_num_clusters))        
        euc_d.fill(-1.0)
        for m in range(0,self.c):
            for l in range(0,self.K[m]):
                euc_d[m,l]   = np.linalg.norm(X-self.mu[m,l])                
                
        return euc_d
        
    
    def predict(self,X,y):
        n_test = len(X)
        count = 0
        
        for i in range(n_test):
            Xi=X[i]
            phi   = self.membershipscore(Xi)
            euc_d = self.euclideanDist(Xi)            
            max_idx_phi=0
            if np.min(phi)>self.C_thr:
                max_idx_phi=np.unravel_index(phi.argmax(),phi.shape)            
            else:
                phi_blended=(1.-self.theta)*phi+self.theta*(self.eps/euc_d)                
                max_idx_phi=np.unravel_index(phi_blended.argmax(),phi_blended.shape)            
                
            test_label =max_idx_phi[0]               
            
            if test_label==int(y[i]):
                count+=1
        
        return 100.*count/n_test
