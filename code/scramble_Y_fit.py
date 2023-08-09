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
    P, _, Yalign, US, U, UB, SB, VB, make_L = setup_data.setup(pfile, yfile, dfile, usfile, ufile, test)

    L, VB, US, SB = make_L(hyperparam[1], hyperparam[2])
    Smat = np.append(np.diag(SB) , np.zeros((UB.shape[0] - len(SB),len(SB))),axis=0)
    Smatinv = np.linalg.pinv(Smat)
    
    lr = LogisticRegression(C = hyperparam[0], class_weight='balanced', penalty = 'l1',solver='liblinear' ,random_state=42, max_iter=1000)
    ix = 0
    while len(glob.glob(outfile + "/*txt.bz2")) < 10: #10000:
        savew = outfile +  str(ix) + ".txt.bz2"
        if os.path.exists(savew):
            continue
        Y_scramb = np.random.permutation(Yalign)
        lr.fit(L, Y_scramb)
        W = lr.coef_.reshape(VB.shape[1],  US.shape[1]).T
        w = pd.DataFrame(W,
                         index=US.columns)
        #w.to_csv("scramb_full/" + fw + "_" + str(i) + ".txt", sep="\t")
        xx = w @ Smatinv @ UB.T
        xx.to_csv(savew, sep="\t") #, compression="gz)
        ix += 1
