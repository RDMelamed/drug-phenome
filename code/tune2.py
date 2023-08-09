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
import setup_data
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
    P, _, Yalign, US, U, UB, SB, VB, make_L = setup_data.setup(pfile, yfile, dfile, usfile, ufile, test)

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

    import pickle
    for us_val in [.8,.9, 1]:
        for p_val in [.8,.9,1]:
            xvsave = outfile + "-US=" + str(us_val) + "-P=" + str(p_val) + "_cv.pkl"
            if os.path.exists(xvsave):
                continue
            L, _, x, _ = make_L(us_val, p_val)
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
        testres[name] = cv['mean_train_score'] ###[cv['split' + str(i) +'_train_score'][0] for i in range(10)]
        params = cv['param_C'].data
    testres = pd.DataFrame(testres, index=params).transpose() #.median(axis=0).sort_values()
    C = testres.mean(axis=0).idxmax()

    ranks = testres[C].idxmax().split("-")
    usrank = float(ranks[0].split("=")[1])
    vbrank = float(ranks[1].split("=")[1])
    lr = LogisticRegression(C = C, class_weight='balanced', penalty = 'l1',solver='liblinear' ,random_state=42, max_iter=1000)
    L, VB_use, US_use, SB_use = make_L(usrank, vbrank)
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
    pd.DataFrame(VB_use,index=P.columns[:VB_use.shape[0]]).to_csv(outfile + "_VB.txt", sep= "\t")
    np.savetxt(outfile + "_SB.txt",SB,delimiter="\t")
    np.savetxt(outfile + "_UB.txt.bz2",UB,delimiter="\t")
    np.savetxt(outfile + "_param.txt",np.array([C, usrank, vbrank]))
        
