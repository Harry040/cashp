import benchmark
import pandas as pd
from sklearn import preprocessing

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





if __name__ == '__main__':

    x,y = benchmark.get_x_y()
    #x = z_score_scale(x)
    print benchmark.get_cross_lg_auc(x,y)



