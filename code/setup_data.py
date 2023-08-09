import pandas as pd
import numpy as np

test_ndrug = 15
test_nphe = 10


def setup(pfile, yfile, dfile, usfile, ufile, test):
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
    
    B = P.values.T
    B = B.T
    UB, SB, VB = np.linalg.svd(B)
    VB = VB.T
    frac_P = (SB/SB.sum()).cumsum()
    
    Yalign2 = Y.transpose().loc[druglist,dislist] #P.columns]
    
    if test:
        Yalign2 = Yalign2.iloc[:test_ndrug, :test_nphe]
        US = US.iloc[:test_ndrug,:]
        VB = VB[:test_nphe,:]
    Yalign = Yalign2.values.flatten("F")
    # Initiliaze Leave One Out CV
    

    def make_L(us_val, p_val, index=[]):
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
        US_use = US
        if len(index) > 0:
            US_use = US_use.iloc[index,:]
        L = np.kron(VB[:,:vb_sel],
                    US_use.iloc[:,:us_sel]) # if not test else np.kron(VB[:test_nphe,:vb_sel], US_use.iloc[:test_ndrug,:us_sel])
        return L, VB[:,:vb_sel], US_use.iloc[:,:us_sel], SB[:vb_sel]
    return P, Yalign2, Yalign, US, U, UB, SB, VB, make_L

def get_id2name():
    names = pd.read_table("input_data/pubchem_names",header=None)
    names[0] = names[0].map(str)
    id2name = names.set_index(0).transpose().loc[1,:].to_dict()
    name2id = names.set_index(1).transpose().loc[0,:].to_dict()
    return id2name, name2id

def load_W_project(saveprefix, US, scramble = False, seed=None):
    id2name, name2id = get_id2name()
    W = pd.read_table(saveprefix,index_col=0)
    if scramble:
        np.random.seed(seed)
        W = pd.DataFrame(np.random.permutation(W.values.reshape(-1,1)).reshape(W.shape))
    U_DS_DdotW = US.iloc[:,:W.shape[0]].dot(W).rename(id2name)
    
    return U_DS_DdotW
