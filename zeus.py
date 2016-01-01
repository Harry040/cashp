import benchmark
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest



def z_score_scale(x, do_dummy=False):


    ft = pd.read_csv('features_type.csv')
    num_col = ft[ft.type == 'numeric']['feature']
    category_col = ft[ft.type == 'category']['feature']


    # numberic feature 
    num_x = x[num_col]
    x_scaled = preprocessing.scale(num_x)


    df_num_x = pd.DataFrame(x_scaled, columns=num_col)
    df_category_x = x[category_col]
    if do_dummy:
        df_catetory_x = pd.get_dummies(df_category_x,columns=category_col)
    df_x = pd.concat([df_num_x, df_category_x], axis=1)
    
    return df_x



def dumplicated(x):
    max = x.max().apply(lambda e: '%.5f'%e)
    min = x.min().apply(lambda e: '%.5f'%e)
    mean = x.mean().apply(lambda e: '%.5f'%e)
    std = x.std().apply(lambda e: '%.5f'%e)
    s = max + min + mean + std
    counts = s.value_counts()
    counts_index = counts.index[counts>1]
    same_values = set(counts_index)
    info = { }
    for i in s.index:
        if s[i] in same_values:
            info.setdefault(s[i],[]).append(i)
    for k in info:
        fs = info[k]
        for i in range(1,len(fs)):
            if all(x[fs[0]] == x[fs[i]]):
                print fs[0], fs[i]
                x.drop(fs[i],axis=1,inplace=1)

    return x

def get_high_corr_features(x):
    corr = x.corr()
    rsv = []
    for i in corr.columns:
        f = corr[i]>0.996
        fs = corr[i].index[f]
        if len(fs)>1:
            fs = fs.drop(i)
            for e in fs:
                rsv.append((i,e))

    new_rsv = []
    fs_set = set()
    for e in rsv:
        if e[0] in fs_set or e[1] in fs_set:
            continue
        new_rsv.append(e)
        fs_set.add(e[0])
        fs_set.add(e[1])

    diff = pd.DataFrame()
    for e in new_rsv:
        f = e[0]+'_'+e[1]
        diff[f] = df[e[0]] - df[e[1]]
    return diff


def get_k_best(x,y, k=300):
    '''
    return k features name
    '''
    sk = SelectKBest(f_classif, k=300)
    sk.fit_transform(x,y)
    return x.columns[sk.get_support()]

if __name__ == '__main__':
    pass
    #x,y = benchmark.get_x_y()
    #x = z_score_scale(x)
    #print benchmark.get_cross_lg_auc(x,y)


