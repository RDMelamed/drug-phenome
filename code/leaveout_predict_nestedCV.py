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
import setup_data
from sklearn.model_selection import GridSearchCV
if __name__ == "__main__":
    pfile = sys.argv[1]
    yfile = sys.argv[2]
    dfile = sys.argv[3]    
    usfile = sys.argv[4]
    ufile = sys.argv[5]
    outfile = sys.argv[6]    
    test = sys.argv[7]=="test"
    hyperparam = np.loadtxt(sys.argv[8])
    startit = int(sys.argv[9])
    endit = int(sys.argv[10])
    P, Ymat, _, US, U, UB, SB, VB, make_L = setup_data.setup(pfile, yfile, dfile, usfile, ufile, test)
    from sklearn.model_selection import KFold
    print("starting with ",yfile) 
    kf = KFold(n_splits=20 if not test else 5)
    Yp = pd.DataFrame(index=Ymat.index, columns=Ymat.columns)
    Ypr = pd.DataFrame(index=Ymat.index, columns=Ymat.columns)
    ofile = outfile + "_" + str(20 if not test else 5) + ".txt"
    if os.path.exists(ofile):
        Yp = pd.read_csv(ofile,sep="\t",index_col=0)
        Ypr = pd.read_csv(outfile + "_" + str(20 if not test else 5) + "_proba.txt",sep="\t",index_col=0)
    
    hyperparam_grid = {"C": [0.1, 1, 10]}
    import pdb
    #pdb.set_trace()
    it = 0
    #with open(outfile + "_log.txt",'w') as f:
    #    f.write("Start " + yfile) 
    import datetime
    for train_ix, test_ix in kf.split(US):
        it +=1
        if it > endit or it < startit:
            continue
        #if os.path.exists(ofile):
        #    Yp = pd.read_csv(ofile,sep="\t",index_col=0)
        #    Ypr = pd.read_csv(outfile + "_" + str(20 if not test else 5) + "_proba.txt",sep="\t",index_col=0)
        
        if pd.isnull(Yp.iloc[test_ix,0]).sum()==0:
            continue
        before_it = datetime.datetime.now()
        L, VB, US, SB = make_L(hyperparam[1], hyperparam[2], train_ix)
        print("it = ",it, " US shape=",US.shape, " VB shape=",VB.shape)
        #'''
        n_split = 10
        n_us = US.shape[0]
        n_vb = VB.shape[0]
        crossval_to_tile = np.tile(np.arange(n_split),int(n_us/n_split)+1)[:n_us]
        full_splits = np.tile(crossval_to_tile, n_vb)
        splits = [[np.where(full_splits!=i)[0], np.where(full_splits==i)[0]] 
                  for i in range(n_split)]

        lr = LogisticRegression(class_weight='balanced', penalty = 'l1',solver='liblinear' ,random_state=42, max_iter=1000)
        grid = GridSearchCV(lr,hyperparam_grid,
                            scoring='f1',
                            cv=splits, 
                            refit=True,
                           verbose=10)

        
        grid.fit(L,Ymat.iloc[train_ix,:].values.flatten("F"))
        C = grid.best_params_['C'] 
        #'''
        #C = 1
        lr = LogisticRegression(C= C, class_weight='balanced', penalty = 'l1',solver='liblinear' ,random_state=42, max_iter=1000)
        with open(outfile + "_log.txt",'a') as f:
            f.write("it =" + str(it)+ "L shape=" + str(L.shape) + " Y shape=" + str( Ymat.iloc[train_ix,:].values.flatten("F").shape) + " Lmax =" + str(L.max().max()) + 
                    " Lmin=" + str(L.min().min())+ " Ymax=" + str(Ymat.iloc[train_ix,:].max().max()) + "\n")
        #L = L[:,:10000]
        lr.fit(L,
               Ymat.iloc[train_ix,:].values.flatten("F"))

        del L
                
 
        L, VB, US, SB = make_L(hyperparam[1], hyperparam[2], test_ix)
        Y_pred = lr.predict(L)
        Yp.iloc[test_ix,:] = np.reshape(Y_pred,(len(test_ix),Ymat.shape[1]),order='F') #.iloc[train_ix,:]
        
        
        Y_pred = lr.predict_proba(L)
        Ypr.iloc[test_ix,:] = np.reshape(Y_pred[:,1],(len(test_ix),Ymat.shape[1]),order='F') #.iloc[train_ix,:]
        Yp.to_csv(outfile + "_" + str(20 if not test else 5) +  "." + str(startit) + "-" + str(endit) +".txt",sep="\t")
        Ypr.to_csv(outfile + "_" + str(20 if not test else 5) + "." + str(startit) + "-" + str(endit) + "_proba.txt",sep="\t")
        after_it = datetime.datetime.now()
        with open(outfile + "_log.txt",'a') as f:
            f.write(str(it) + "C="+str( C) + "time = " + '{:1.2f}'.format((after_it - before_it).total_seconds()/60/60) + "\n")
    print("finished")
