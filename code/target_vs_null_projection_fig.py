import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def violin_per_target(t2d,height,  drug_phenome_distances, drug_scr_distances):
    real = drug_phenome_distances.reset_index()
    scr = drug_scr_distances.reset_index()

    pos = []
    lab = []
    reals = []
    scrs = []
    xpos = 0
    diffs = []
    for targ,drug in t2d.items(): #tct.loc[(tct['ct'] > 2),:].index:
        if len(drug) <3:
            continue
        r = real.loc[real['level_0'].isin(drug) & real['level_1'].isin(drug),0].values
        reals.append(r)
        s = scr.loc[scr['level_0'].isin(drug) & scr['level_1'].isin(drug),0].values
        scrs.append(s)
        pos.append(xpos)
        #pos.append(ix)
        lab.append(targ + " (" + str(len(drug)) + ")")
        xpos += 1
        diffs.append(stats.ranksums(r,s)[1])
    sort = np.argsort(np.array(diffs))[::-1]
    realS = [reals[i] for i in sort]
    scrS = [scrs[i] for i in sort]
    labS = [lab[i] for i in sort]
    diffS = [diffs[i] for i in sort]


    def targviolin(reals, pos, ax,body, line):
        parts = ax.violinplot(reals, positions=pos, vert=False, showmedians=True)

        for pc in parts['bodies']:
            pc.set_facecolor(body)
            pc.set_edgecolor(body)
            pc.set_alpha(.5)
        for partname in ('cbars','cmins','cmaxes','cmedians'):
            vp = parts[partname]
            vp.set_edgecolor(line)
            vp.set_linewidth(1)

    f,ax = plt.subplots(1,figsize=(3,height))
    targviolin(realS, pos, ax, 'red','red')
    targviolin(scrS, pos, ax, 'blue','blue')
    ax.set_yticks(pos)
    ax.set_yticklabels(labS)
    ax.text( -.4,max(pos)+1.1,'pairwise distance in phenome effect matrix',color='red')
    ax.text( -.4,max(pos)+.5,'pairwise distance in null model',color='blue')
    ax.set_ylim(-.5, max(pos) + 1.2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for ix, d in enumerate(diffS):
        signif = int(-1*np.log10(d))
        if signif < 1:
            continue
        ax.text(-.2, pos[ix],'*'*min(signif,5) + ("+" if signif > 5 else ''))
    #f.savefig("fig2C.tiff",bbox_inches='tight',dpi=300)


def target_pairwise_in_vs_out(uu, t2drugs,d2t):
    drug_remove = list(set(uu.index.levels[0]) - set(d2t.keys())) + [d for d,ts in d2t.items() if len(ts)==0]

    tt = {}
    gkeep = (uu.max(axis=0) - uu.min(axis=0) ) > 0
    for targ,drugs in t2drugs.items(): #tct.loc[(tct['ct'] > 2),:].index:
        if len(drugs) < 3:
            continue
        compvec = []        
        for d1, d2 in uu.index:
            if d1 in drug_remove or d2 in drug_remove:
                compvec.append(-1)
                continue

            if d1 in drugs and d2 in drugs:
                compvec.append(1)
            elif (d1 in drugs or d2 in drugs) and (d1 in d2t and d2 in d2t):
                ## then compare to sim of d1 or d2 to a drug NOT sharing a target
                if len(set(d2t[d1]) & set(d2t[d2]) ) == 0:
                    compvec.append(0)
                else:
                    compvec.append(-1)
            else:
                    compvec.append(-1)                
        compvec = np.array(compvec)
        if sum(compvec==0)==0 or sum(compvec==1)==0:
            print(targ)
        tt[targ] = [len(drugs)] + list(stats.ranksums(uu.loc[compvec==0],uu.loc[compvec==1])) +[uu.loc[compvec==0].median(),uu.loc[compvec==1].median()]
    return pd.DataFrame(tt, index = ['n','t','p','omed','imed']).transpose().sort_values('p')
    
