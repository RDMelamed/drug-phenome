import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pickle
import os
import pdb
import sys
import glob
from sklearn.model_selection import GridSearchCV
test_ndrug = 12
test_nphe = 10
if __name__=="__main__":

    pfile = sys.argv[1]
    yfile = sys.argv[2]
    dfile = sys.argv[3]    
    usfile = sys.argv[4]
    ufile = sys.argv[5]
    outfile = sys.argv[6]    
    test = sys.argv[7]=="test"

    P = pd.read_csv(pfile, sep = '\t', index_col = 0)
    Y = pd.read_csv(yfile, sep="\t",  index_col=0)

    D = pd.read_csv(dfile, sep='\t',index_col=0)
    print('usfile is ',usfile)
    US = np.loadtxt(usfile)
    #pdb.set_trace()
    US = pd.DataFrame(US, index = D.columns)
    U = np.loadtxt(ufile)
    sq = np.matmul( np.linalg.pinv(U), US.values)
    #S = pd.Series(np.diag(sq))
    frac_US = (np.diag(sq)/np.diag(sq).sum()).cumsum()
    #frac_US = (sq/sq.sum()).cumsum()

    del D
    druglist = set(US.index) & set(Y.columns)
    dislist = set(P.columns) & set(Y.index)
    US = US.loc[druglist,:]
    P = P.loc[:,dislist]
    Yalign2 = Y.transpose().loc[druglist,dislist] #P.columns]
    if test:
        Yalign2 = Yalign2.iloc[:test_ndrug, :test_nphe]
    Yalign = Yalign2.values.flatten("F")
    B = P.values.T
    B = B.T
    UB, SB, VB = np.linalg.svd(B)
    VB = VB.T
    frac_P = (SB/SB.sum()).cumsum()
    # Initiliaze Leave One Out CV
    
    hyperparam_grid = {"C": [0.1, 1, 10]}
    lr = LogisticRegression(class_weight='balanced', penalty = 'l1',
                                 random_state=42, max_iter=5000, solver = 'liblinear')

    n_split = 10
    n_us = US.shape[0]
    n_vb = VB.shape[0]
    if test:
        n_us = test_ndrug
        n_vb = test_nphe
    crossval_to_tile = np.tile(np.arange(n_split),int(n_us/n_split)+1)[:n_us]
    full_splits = np.tile(crossval_to_tile, n_vb)
    splits = [[np.where(full_splits!=i)[0], np.where(full_splits==i)[0]] 
              for i in range(n_split)]


    grid = GridSearchCV(lr,hyperparam_grid,
                        scoring='f1',
                        cv=splits, 
                        refit=True,
                       verbose=10)
    def make_L(us_val, p_val):
        us_sel = frac_US >= us_val
        if sum(us_sel > 0):
            us_sel = np.where(us_sel)[0][0]
        else:
            us_sel = US.shape[1]
        vb_sel = frac_P >= p_val
        if sum(vb_sel > 0):
            vb_sel = np.where(vb_sel)[0][0]
        else:
            vb_sel = VB.shape[1]        
        L = np.kron(VB[:,:vb_sel], US.iloc[:,:us_sel]) if not test else np.kron(VB[:test_nphe,:vb_sel], US.iloc[:test_ndrug,:us_sel])
        return L, VB[:,:vb_sel], US.iloc[:,:us_sel]
    
    import pickle
    for us_val in [.8,.9, 1]:
        for p_val in [.8,.9,1]:
            xvsave = outfile + "-US=" + str(us_val) + "-P=" + str(p_val) + "_cv.pkl"
            if os.path.exists(xvsave):
                continue
            L, _, x = make_L(us_val, p_val)
            grid.fit(L,Yalign)
            #print("US:", us_val, us_sel)
            #print("VB:", p_val, vb_sel)        
            with open(xvsave,'wb') as f:
                pickle.dump(grid.cv_results_, f)

    testres = {}
    #trainres = {}
    params = []
    for cvf in glob.glob(outfile + "*_cv.pkl"):
        cv = pickle.load(open(cvf,'rb'))
        name = "-".join(os.path.basename(cvf).split("-")[1:]).split("_")[0]
        #testres[name] = [cv['split' + str(i) +'_test_score'][0] for i in range(10)]
        testres[name] = cv['mean_test_score'] ###[cv['split' + str(i) +'_train_score'][0] for i in range(10)]
        params = cv['param_C'].data
    testres = pd.DataFrame(testres, index=params).transpose() #.median(axis=0).sort_values()
    C = testres.mean(axis=0).idxmax()

    ranks = testres[C].idxmax().split("-")
    usrank = float(ranks[0].split("=")[1])
    vbrank = float(ranks[1].split("=")[1])
    lr = LogisticRegression(C = C, class_weight='balanced', penalty = 'l1',solver='liblinear' ,random_state=42, max_iter=1000)
    L, VB_use, US_use = make_L(usrank, vbrank)
    lr.fit(L, Yalign)
    W = lr.coef_.reshape(VB_use.shape[1], US_use.shape[1]).T
    pickle.dump(lr, open(outfile + "_lr.pkl",'wb'))
    #np.savetxt(fw, W,delimiter="\t")
    #pdb.set_trace()    
    w = pd.DataFrame(W,
                     index=US_use.columns)
    #pd.DataFrame(P,index=P.columns).to_csv(fw + "_VB.txt",sep="\t")
    w.to_csv(outfile + "_W.txt",sep="\t")
    pd.DataFrame(P,index=P.columns).to_csv(outfile + "_VB.txt",sep="\t")

    US_use.to_csv(outfile + "_US.txt", sep= "\t")
    pd.DataFrame(VB_use,index=P.columns).to_csv(outfile + "_VB.txt", sep= "\t")
    np.savetxt(outfile + "_SB.txt",SB,delimiter="\t")
    np.savetxt(outfile + "_UB.txt.bz2",UB,delimiter="\t", compression='bz2')
    np.savetxt(outfile + "_param.txt",np.array([C, usrank, vbrank]))
        
