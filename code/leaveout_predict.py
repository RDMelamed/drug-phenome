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
if __name__ == "__main__":
    pfile = sys.argv[1]
    yfile = sys.argv[2]
    dfile = sys.argv[3]    
    usfile = sys.argv[4]
    ufile = sys.argv[5]
    outfile = sys.argv[6]    
    test = sys.argv[7]=="test"
    hyperparam = np.loadtxt(sys.argv[8])
    P, Ymat, _, US, U, UB, SB, VB, make_L = setup_data.setup(pfile, yfile, dfile, usfile, ufile, test)
    from sklearn.model_selection import KFold
    lr = LogisticRegression(C = hyperparam[0], class_weight='balanced', penalty = 'l1',solver='liblinear' ,random_state=42, max_iter=1000)
    kf = KFold(n_splits=20 if not test else 5)
    Yp = pd.DataFrame(index=Ymat.index, columns=Ymat.columns)
    for train_ix, test_ix in kf.split(US):

        L, VB, US, SB = make_L(hyperparam[1], hyperparam[2], train_ix)
        lr.fit(L,
               Ymat.iloc[train_ix,:].values.flatten("F"))
        del L
        L, VB, US, SB = make_L(hyperparam[1], hyperparam[2], test_ix)
        Y_pred = lr.predict(L)
        Yp.iloc[test_ix,:] = np.reshape(Y_pred,(len(test_ix),Ymat.shape[1]),order='F')
    Yp.to_csv(outfile + "_" + str(20 if not test else 5) + ".txt",sep="\t")
