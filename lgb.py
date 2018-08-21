# coding: UTF-8
import pandas as pd
import numpy as np
import time
import lightgbm as lgb
import xgboost as xgb
import gc
file_dirs = '/mnt/datasets/fusai/'

ori_dirs = './'

user_reg = pd.read_csv(file_dirs + '/user_register_log.txt', sep='\t', header=None,
                                   names=['user_id', 'register_day', 'register_type', 'device_type'])
user_reg.to_csv(ori_dirs+'user_reg.csv', index=False)

app = pd.read_csv(file_dirs + '/app_launch_log.txt', sep='\t', header=None, names=['user_id', 'app_launch_day'])
app.to_csv(ori_dirs+'app.csv', index=False)

vedio = pd.read_csv(file_dirs + '/video_create_log.txt', sep='\t', header=None,
                                  names=['user_id', 'video_create_day'])
vedio.to_csv(ori_dirs+'vedio.csv', index=False)

user_act = pd.read_csv(file_dirs + '/user_activity_log.txt', sep='\t', header=None,
                                   names=['user_id', 'user_activity_day', 'page', 'video_id', 'author_id', 'action_type'])
user_act.to_csv(ori_dirs+'user_act.csv', index=False)
del user_reg, app, vedio, user_act
print('finished!')
gc.collect()

# coding: UTF-8
import pandas as pd

data_file = './'
all_dataSet_path = data_file+'all_dataSet.csv'
first_train_path = data_file+'first_train_'
first_test_path = data_file+'first_test_'
sec_train_path = data_file+'sec_train_'
sec_test_path = data_file+'sec_test_'
thr_dataSet_path = data_file+'thr_train_'

# 1-16的数据 预测17-23某用户是否活跃
# 8-23的数据 预测24-30某用户是否活跃

# 测试集
# 以15-30数据 预测31-37某用户是否活跃

app = pd.read_csv('./app.csv')
vedio = pd.read_csv('./vedio.csv')
user_reg = pd.read_csv('./user_reg.csv')
user_act = pd.read_csv('./user_act.csv')


def cut_data_as_time(dataSet_path, new_dataSet_path, begin_day, end_day):
    temp_user_reg = user_reg[(user_reg['register_day'] >= begin_day) & (user_reg['register_day'] <= end_day)]
    temp_user_act = user_act[(user_act['user_activity_day'] >= begin_day) & (user_act['user_activity_day'] <= end_day)]
    temp_vedio = vedio[(vedio['video_create_day'] >= begin_day) & (vedio['video_create_day'] <= end_day)]
    temp_app = app[(app['app_launch_day'] >= begin_day) & (app['app_launch_day'] <= end_day)]

    temp_user_reg.to_csv(new_dataSet_path+'user_reg.csv', index=False)
    temp_user_act.to_csv(new_dataSet_path+'user_act.csv', index=False)
    temp_vedio.to_csv(new_dataSet_path+'vedio.csv', index=False)
    temp_app.to_csv(new_dataSet_path+'app.csv', index=False)
    del temp_user_reg, temp_user_act, temp_vedio, temp_app
    print('数据切分完成', new_dataSet_path)


def generate_dataSet():
    begin_day = 1
    end_day = 16
    cut_data_as_time(all_dataSet_path, first_train_path, begin_day, end_day)
    begin_day = 17
    end_day = 23
    cut_data_as_time(all_dataSet_path, first_test_path, begin_day, end_day)
    print ('第一个数据集产生成功....')

    begin_day = 8
    end_day = 23
    cut_data_as_time(all_dataSet_path, sec_train_path, begin_day, end_day)
    begin_day = 24
    end_day = 30
    cut_data_as_time(all_dataSet_path, sec_test_path, begin_day, end_day)
    print ('第二个数据集产生成功....')

    begin_day = 15
    end_day = 30
    cut_data_as_time(all_dataSet_path, thr_dataSet_path, begin_day, end_day)
    print ('预测训练集生成成功....')

generate_dataSet()

# coding: UTF-8
import time
import pandas as pd
import numpy as np

data_file = './'
first_train_path = data_file+'first_train_'
first_test_path = data_file+'first_test_'
sec_train_path = data_file+'sec_train_'
sec_test_path = data_file+'sec_test_'
thr_dataSet_path = data_file+'thr_train_'

train_path = './train.csv'
val_path = './val.csv'
test_path = './test.csv'
app = 'app.csv'
vedio = 'vedio.csv'
user_reg = 'user_reg.csv'
user_act = 'user_act.csv'


# 构造训练集和测试集与特征
# 获取所有id 查看对应id是否在测试集里面出现过
def get_train_label(train_path, test_path, end):
    train_reg = pd.read_csv('./user_reg.csv')
    train_reg = train_reg[train_reg.register_day <= end]
    train_data_id = np.unique(train_reg['user_id'])

    test_cre = pd.read_csv(test_path+vedio, usecols=['user_id'])
    test_lau = pd.read_csv(test_path+app, usecols=['user_id'])
    test_act = pd.read_csv(test_path+user_act, usecols=['user_id'])
    test_data_id = np.unique(pd.concat([test_cre, test_lau, test_act]))

    train_label = []
    for i in train_data_id:
        if i in test_data_id:
            train_label.append(1)
        else:
            train_label.append(0)
    train_data = pd.DataFrame()
    train_data['user_id'] = train_data_id
    train_data['label'] = train_label
    return train_data


def get_test(test_path):
    test_reg = pd.read_csv('./user_reg.csv')
    test_data_id = np.unique(test_reg['user_id'])

    test_data = pd.DataFrame()
    test_data['user_id'] = test_data_id
    return test_data


def deal_feature(path, user_id, ls_date):
    reg = pd.read_csv(path+user_reg)
    cre = pd.read_csv(path+vedio)
    lau = pd.read_csv(path+app)
    act = pd.read_csv(path+user_act)
    feature = pd.DataFrame()
    feature['user_id'] = user_id
    print('changdu', len(feature['user_id']))
    
    def get_max_lx(array):
        array=np.array(array)
        array.sort()
        k=np.diff(array)
        index=len(k)
        i=0
        days=0
        m=0
        while(i<=index-1):
            if(k[i]==1):
                days+=1
                i+=1
                m=max(days,m)
                if(i==index):
                    break
            else:
                days=0
                i+=1
                if(i==index):
                    break
        return m+1
    def get_last_lx(a):
        a=np.array(a)
        a.sort()
        k=np.diff(a)
        index=len(k)
        i=0
        days=0
        m=0
        result=[]
        while(i<=index-1):
            while((k[i]==1)&(i<=index-1)):
                days=days+1
                i=i+1
                m=days
                if(i==index):
                    break
            if(m>0):
                result.append(m+1)
            m=0
            days=0
            i+=1
        last=len(result)-1
        if(last>=0):
            return result[last]
        else:
            return 1
    def get_lx_var(a):
        a=np.array(a)
        a.sort()
        k=np.diff(a)
        index=len(k)
        i=0
        days=0
        m=0
        result=[]
        while(i<=index-1):
            while((k[i]==1)&(i<=index-1)):
                days=days+1
                i=i+1
                m=days
                if(i==index):
                    break
            if (m > 0):
                result.append(m+1)
            m = 0
            days = 0
            i += 1
        r = np.array(result)
        if (len(r) <= 1):
            return 0
        else:
            var = r.var()
            return var

    ########################################################## create表 ##########################################################
    s = time.time()
    # 拍摄时间度量
    result = cre.groupby(['user_id'], as_index=False)['video_create_day'].agg({
        'create_day_count': 'count',
        'create_day_max': 'max',
        'create_day_min': 'min',
        'create_day_mean': 'mean',
        # 'create_day_median': 'median',
        # 'create_day_skew': 'skew',
        # 'create_day_std': 'std',
        'create_day_last': 'last',
    })
    feature = pd.merge(feature, result, on='user_id', how='left')
    # 创建天数unqiue,ratio
    cre_feature = cre.groupby('user_id')['video_create_day'].nunique().reset_index()
    cre_feature.columns = ['user_id', 'create_day_uni']
    feature = pd.merge(feature, cre_feature, on='user_id', how='left')
    feature['create_day_radio'] = feature['create_day_uni']/feature['create_day_count']
    # 最后创建的时间和最大注册时间的差值
    feature['create_register_day'] = feature['create_day_last'] - reg['register_day'].max()
    feature['create_register_day_min'] = feature['create_day_min'] - cre['video_create_day'].min()
    feature['create_register_day_mean'] = feature['create_day_mean'] - cre['video_create_day'].mean()
    
    # *************************zuihouchuangjianshangjianshijian
    last_video_time = cre.sort_values('video_create_day').drop_duplicates(subset='user_id', keep='last')
    last_video_time['last_video_time'] = ls_date - last_video_time['video_create_day']
    feature = feature.merge(last_video_time[['last_video_time', 'user_id']], 'left', on='user_id')
    # ****************************video 总数
    video_count_sum = cre.groupby('user_id').size().rename('video_count_sum').reset_index()
    feature = feature.merge(video_count_sum, 'left', on='user_id')
    
    # 拍摄视频时间差度量
    cre_feature = cre.groupby('user_id')['video_create_day'].diff().fillna(0).reset_index(name='create_diff')
    cre['create_diff'] = cre_feature['create_diff']
    result = cre.groupby(['user_id'], as_index=False)['create_diff'].agg({
        'create_day_diff_max': 'max',
        'create_day_diff_min': 'min',
        'create_day_diff_mean': 'mean',
        # 'create_day_diff_median': 'median',
        'create_day_diff_ske': 'skew',
        # 'create_day_diff_std': 'std',
        'create_day_diff_last': 'last',
    })
    feature = pd.merge(feature, result, on='user_id', how='left')
    # 拍摄视频时间差峰度kurt
    # kurt = cre.groupby(['user_id'])['create_diff'].apply(pd.DataFrame.kurt).reset_index(name='create_day_diff_kur')
    # feature = pd.merge(feature, kurt, on='user_id', how='left')
    
    # 每天件视频的数量特征
    video_day_sum = cre.groupby(['user_id', 'video_create_day']).size().rename('video_day_sum').reset_index()
    video_day_sum_var = video_day_sum.groupby('user_id')['video_day_sum'].var().rename('video_day_sum_var').reset_index()
    feature = feature.merge(video_day_sum_var, 'left', on='user_id')
    
    # rank的特征工程
    cre['day_rank'] = cre.groupby('user_id')['video_create_day'].rank(ascending=False, method='dense')
    result = cre.groupby(['user_id'], as_index=False)['day_rank'].agg({
        'rank_create_day_count': 'count',
        'rank_create_day_max': 'max',
        'rank_create_day_min': 'min',
        'rank_create_day_mean': 'mean',
        'rank_create_day_median': 'median',
        'rank_create_day_skew': 'skew',
        'rank_create_day_std': 'std',
        'rank_create_day_last': 'last',
    })
    feature = pd.merge(feature, result, on='user_id', how='left')
    
    # 添加时间连续特征
    cre_max_lx_days = cre.groupby('user_id').agg({'video_create_day': get_max_lx}).reset_index().rename(columns={'video_create_day': 'cre_max_lx_days'})
    cre_last_lx_days = cre.groupby('user_id').agg({'video_create_day': get_last_lx}).reset_index().rename(columns={'video_create_day': 'cre_last_lx_days'})
    cre_lx_var = cre.groupby('user_id').agg({'video_create_day': get_lx_var}).reset_index().rename(columns={'video_create_day': 'cre_lx_var'})
    feature = pd.merge(feature, cre_max_lx_days, on='user_id', how='left')
    feature = pd.merge(feature, cre_last_lx_days, on='user_id', how='left')
    feature = pd.merge(feature, cre_lx_var, on='user_id', how='left')
    
    print('create表特征提取完毕',int(time.time()-s),'s')
    print('changdu', len(feature['user_id']))

    # 距离最后天数的特征集
    cre1 = cre[cre['video_create_day'] == (ls_date-1)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_1_create_cout'})
    cre2 = cre[cre['video_create_day'] == (ls_date-2)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_2_create_cout'})
    cre3 = cre[cre['video_create_day'] == (ls_date-3)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_3_create_cout'})
    cre4 = cre[cre['video_create_day'] == (ls_date-4)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_4_create_cout'})
    cre5 = cre[cre['video_create_day'] == (ls_date-5)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_5_create_cout'})
    cre6 = cre[cre['video_create_day'] == (ls_date-6)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_6_create_cout'})
    feature = pd.merge(feature, cre1, on='user_id', how='left')
    feature = pd.merge(feature, cre2, on='user_id', how='left')
    feature = pd.merge(feature, cre3, on='user_id', how='left')
    feature = pd.merge(feature, cre4, on='user_id', how='left')
    feature = pd.merge(feature, cre5, on='user_id', how='left')
    feature = pd.merge(feature, cre6, on='user_id', how='left')
    
    # 添加时间划窗的特征3,5,7天
    cre_day_3 = cre[(cre['video_create_day']<=ls_date) & (cre['video_create_day']>ls_date-3)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_day3_create_cout'})
    cre_day_3['ls_day3_create_cout'] = 0.7*cre_day_3['ls_day3_create_cout']
    feature = pd.merge(feature, cre_day_3, on='user_id', how='left')
    cre_day_5 = cre[(cre['video_create_day']<=ls_date) & (cre['video_create_day']>ls_date-5)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_day5_create_cout'})
    cre_day_5['ls_day5_create_cout'] = 0.3*cre_day_5['ls_day5_create_cout']
    feature = pd.merge(feature, cre_day_5, on='user_id', how='left')
    cre_day_7 = cre[(cre['video_create_day']<=ls_date) & (cre['video_create_day']>ls_date-7)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_day7_create_cout'})
    cre_day_7['ls_day7_create_cout'] = 0.1*cre_day_7['ls_day7_create_cout']
    feature = pd.merge(feature, cre_day_7, on='user_id', how='left')

    ########################################################## register表 ##########################################################
    s = time.time()
    feature = pd.merge(feature, reg, on='user_id', how='left')
    # 注册时间度量
    result = reg.groupby(['user_id'], as_index=False)['register_day'].agg({
        'register_day_count': 'count',
        'register_day_max': 'max',
        'register_day_min': 'min',
        'register_day_mean': 'mean',
        # 'register_day_median': 'median',
        # 'register_day_skew': 'skew',
        # 'register_day_std': 'std',
        'register_day_last': 'last',
    })
    feature = pd.merge(feature, result, on='user_id', how='left')
    # 创建天数unqiue,ratio
    reg_feature = reg.groupby('user_id')['register_day'].nunique().reset_index()
    reg_feature.columns = ['user_id', 'register_day_uni']
    feature = pd.merge(feature, reg_feature, on='user_id', how='left')
    feature['register_day_radio'] = feature['register_day_uni']/feature['register_day_count']
    # 最后创建的时间和最大注册时间的差值
    feature['register_register_day'] = feature['register_day_last'] - reg['register_day'].max()
    
    # 距离最后天数的特征集
    reg1 = reg[reg['register_day'] == (ls_date-1)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_1_greister_cout'})
    reg2 = reg[reg['register_day'] == (ls_date-2)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_2_greister_cout'})
    reg3 = reg[reg['register_day'] == (ls_date-3)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_3_greister_cout'})
    reg4 = reg[reg['register_day'] == (ls_date-4)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_4_greister_cout'})
    reg5 = reg[reg['register_day'] == (ls_date-5)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_5_greister_cout'})
    reg6 = reg[reg['register_day'] == (ls_date-6)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_6_greister_cout'})
    feature = pd.merge(feature, reg1, on='user_id', how='left')
    feature = pd.merge(feature, reg2, on='user_id', how='left')
    feature = pd.merge(feature, reg3, on='user_id', how='left')
    feature = pd.merge(feature, reg4, on='user_id', how='left')
    feature = pd.merge(feature, reg5, on='user_id', how='left')
    feature = pd.merge(feature, reg6, on='user_id', how='left')
    
    # 添加时间划窗的特征3,5,7天
    reg_day_3 = reg[(reg['register_day']<=ls_date) & (reg['register_day']>ls_date-3)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_day3_register_cout'})
    reg_day_3['ls_day3_register_cout'] = 0.7*reg_day_3['ls_day3_register_cout']
    feature = pd.merge(feature, reg_day_3, on='user_id', how='left')
    reg_day_5 = reg[(reg['register_day']<=ls_date) & (reg['register_day']>ls_date-5)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_day5_register_cout'})
    reg_day_5['ls_day5_register_cout'] = 0.3*reg_day_5['ls_day5_register_cout']
    feature = pd.merge(feature, reg_day_5, on='user_id', how='left')
    reg_day_7 = reg[(reg['register_day']<=ls_date) & (reg['register_day']>ls_date-7)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_day7_register_cout'})
    reg_day_7['ls_day7_register_cout'] = 0.1*reg_day_7['ls_day7_register_cout']
    feature = pd.merge(feature, reg_day_7, on='user_id', how='left')
    
     # 添加时间连续特征
    reg_max_lx_days = reg.groupby('user_id').agg({'register_day': get_max_lx}).reset_index().rename(columns={'register_day': 'reg_max_lx_days'})
    reg_last_lx_days = reg.groupby('user_id').agg({'register_day': get_last_lx}).reset_index().rename(columns={'register_day': 'reg_last_lx_days'})
    reg_lx_var = reg.groupby('user_id').agg({'register_day': get_lx_var}).reset_index().rename(columns={'register_day': 'reg_lx_var'})
    feature = pd.merge(feature, reg_max_lx_days, on='user_id', how='left')
    feature = pd.merge(feature, reg_last_lx_days, on='user_id', how='left')
    feature = pd.merge(feature, reg_lx_var, on='user_id', how='left')
    print('register表特征提取完毕',int(time.time()-s),'s')
    print('changdu', len(feature['user_id']))

    ########################################################## launch表 ##########################################################
    s = time.time()
    # 拍摄时间度量
    result = lau.groupby(['user_id'], as_index=False)['app_launch_day'].agg({
        'launch_day_count': 'count',
        'launch_day_max': 'max',
        'launch_day_min': 'min',
        'launch_day_mean': 'mean',
        # 'launch_day_median' : 'median',
        # 'launch_day_skew'   : 'skew',
        # 'launch_day_std'    : 'std',
        'launch_day_last': 'last',
    })
    feature = pd.merge(feature, result, on='user_id', how='left')
    # 创建天数unqiue,ratio
    lau_feature = lau.groupby('user_id')['app_launch_day'].nunique().reset_index()
    lau_feature.columns = ['user_id', 'launch_day_uni']
    feature = pd.merge(feature, lau_feature, on='user_id', how='left')
    feature['launch_day_radio'] = feature['launch_day_uni']/feature['launch_day_count']
    # 最后创建的时间和最大注册时间的差值
    feature['launch_register_day'] = feature['launch_day_last'] - reg['register_day'].max()
    feature['launch_register_day_mean'] = feature['launch_day_mean'] - lau['app_launch_day'].mean()
    # ***********************距离结束的时间差
    last_login_data = lau.sort_values('app_launch_day').drop_duplicates(subset='user_id', keep='last')
    last_login_data['last_login_time'] = ls_date - last_login_data['app_launch_day']
    feature = feature.merge(last_login_data[['last_login_time', 'user_id']], 'left', on='user_id')
    # ***********************查看总数
    login_count_sum_ = lau.groupby('user_id').size().rename('login_count_sum').reset_index()
    feature = feature.merge(login_count_sum_, 'left', on='user_id')
    # 拍摄视频时间差度量
    lau_feature = lau.groupby('user_id')['app_launch_day'].diff().fillna(0).reset_index(name='launch_diff')
    lau['launch_diff'] = lau_feature['launch_diff']
    result = lau.groupby(['user_id'], as_index=False)['launch_diff'].agg({
        'launch_day_diff_max': 'max',
        'launch_day_diff_min': 'min',
        'launch_day_diff_mean': 'mean',
        # 'launch_day_diff_median'  : 'median',
        'launch_day_diff_ske': 'skew',
        # 'launch_day_diff_std'     : 'std',
        'launch_day_diff_last': 'last',
    })
    feature = pd.merge(feature, result, on='user_id', how='left')
    # 拍摄视频时间差峰度kurt
    # kurt = lau.groupby(['user_id'])['launch_diff'].apply(pd.DataFrame.kurt).reset_index(name='launch_day_diff_kur')
    # feature = pd.merge(feature, kurt, on='user_id', how='left')
    
    # 每天登陆app的数量特征
    lau_day_sum = lau.groupby(['user_id', 'app_launch_day']).size().rename('lau_day_sum').reset_index()
    lau_day_sum_var = lau_day_sum.groupby('user_id')['lau_day_sum'].var().rename('lau_day_sum_var').reset_index()
    feature = feature.merge(lau_day_sum_var, 'left', on='user_id')
    
    # rank的特征工程
    lau['day_rank'] = lau.groupby('user_id')['app_launch_day'].rank(ascending=False, method='dense')
    result = lau.groupby(['user_id'], as_index=False)['day_rank'].agg({
        'rank_launch_day_count': 'count',
        'rank_launch_day_max': 'max',
        'rank_launch_day_min': 'min',
        'rank_launch_day_mean': 'mean',
        'rank_launch_day_median': 'median',
        'rank_launch_day_skew': 'skew',
        'rank_launch_day_std': 'std',
        'rank_launch_day_last': 'last',
    })
    feature = pd.merge(feature, result, on='user_id', how='left')
    
    # 添加时间连续特征
    lau_max_lx_days = lau.groupby('user_id').agg({'app_launch_day': get_max_lx}).reset_index().rename(columns={'app_launch_day': 'lau_max_lx_days'})
    lau_last_lx_days = lau.groupby('user_id').agg({'app_launch_day': get_last_lx}).reset_index().rename(columns={'app_launch_day': 'lau_last_lx_days'})
    lau_lx_var = lau.groupby('user_id').agg({'app_launch_day': get_lx_var}).reset_index().rename(columns={'app_launch_day': 'lau_lx_var'})
    feature = pd.merge(feature, lau_max_lx_days, on='user_id', how='left')
    feature = pd.merge(feature, lau_last_lx_days, on='user_id', how='left')
    feature = pd.merge(feature, lau_lx_var, on='user_id', how='left')
    
    print('launch表特征提取完毕',int(time.time()-s),'s')
    print('changdu', len(feature['user_id']))

    # 距离最后天数的特征集
    lau1 = lau[lau['app_launch_day'] == (ls_date-1)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_1_launch_cout'})
    lau2 = lau[lau['app_launch_day'] == (ls_date-2)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_2_launch_cout'})
    lau3 = lau[lau['app_launch_day'] == (ls_date-3)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_3_launch_cout'})
    lau4 = lau[lau['app_launch_day'] == (ls_date-4)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_4_launch_cout'})
    lau5 = lau[lau['app_launch_day'] == (ls_date-5)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_5_launch_cout'})
    lau6 = lau[lau['app_launch_day'] == (ls_date-6)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_6_launch_cout'})
    feature = pd.merge(feature, lau1, on='user_id', how='left')
    feature = pd.merge(feature, lau2, on='user_id', how='left')
    feature = pd.merge(feature, lau3, on='user_id', how='left')
    feature = pd.merge(feature, lau4, on='user_id', how='left')
    feature = pd.merge(feature, lau5, on='user_id', how='left')
    feature = pd.merge(feature, lau6, on='user_id', how='left')
    
     # 添加时间划窗的特征3,5,7天
    lau_day_3 = lau[(lau['app_launch_day']<=ls_date) & (lau['app_launch_day']>ls_date-3)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_day3_launch_cout'})
    lau_day_3['ls_day3_launch_cout'] = 0.7*lau_day_3['ls_day3_launch_cout']
    feature = pd.merge(feature, lau_day_3, on='user_id', how='left')
    lau_day_5 = lau[(lau['app_launch_day']<=ls_date) & (lau['app_launch_day']>ls_date-5)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_day5_launch_cout'})
    lau_day_5['ls_day5_launch_cout'] = 0.3*lau_day_5['ls_day5_launch_cout']
    feature = pd.merge(feature, lau_day_5, on='user_id', how='left')
    lau_day_7 = lau[(lau['app_launch_day']<=ls_date) & (lau['app_launch_day']>ls_date-7)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_day7_launch_cout'})
    lau_day_7['ls_day7_launch_cout'] = 0.1*lau_day_7['ls_day7_launch_cout']
    feature = pd.merge(feature, lau_day_7, on='user_id', how='left')

    ########################################################## activity表 ##########################################################
    s = time.time()
    # 行为时间度量
    result = act.groupby(['user_id'], as_index=False)['user_activity_day'].agg({
        'activity_day_count': 'count',
        'activity_day_max': 'max',
        'activity_day_min': 'min',
        'activity_day_mean': 'mean',
        # 'activity_day_median': 'median',
        # 'activity_day_skew': 'skew',
        # 'activity_day_std': 'std',
        'activity_day_last': 'last',
    })
    feature = pd.merge(feature, result, on='user_id', how='left')
    # 行为的总数
    act_count_cnt = act.groupby('user_id').size().rename('act_count_sum').reset_index()
    feature = pd.merge(feature, act_count_cnt, on='user_id', how='left')
    # 行为天数unqiue,ratio
    act_feature = act.groupby('user_id')['user_activity_day'].nunique().reset_index()
    act_feature.columns = ['user_id', 'activity_day_uni']
    feature = pd.merge(feature, act_feature, on='user_id', how='left')
    feature['activity_day_radio'] = feature['activity_day_uni']/feature['activity_day_count']
    # 最后行为的时间与最大注册时间的差值
    feature['activity_register_day'] = feature['activity_day_last'] - reg['register_day'].max()
    feature['activity_register_day_min'] = feature['activity_day_min'] - act['user_activity_day'].min()
    # 首次行为时间与最大注册时间的差值
    feature['activity_register_day_first'] = feature['activity_day_min'] - reg['register_day'].max()
    # ******************zuihoudenglu
    last_act_time = act.sort_values('user_activity_day').drop_duplicates(subset='user_id', keep='last')
    last_act_time['last_act_time'] = ls_date - last_act_time['user_activity_day']
    feature = feature.merge(last_act_time[['last_act_time', 'user_id']], 'left', on='user_id')
    # unique
    act_feature = act.groupby('user_id')['video_id'].nunique().reset_index()
    act_feature.columns = ['user_id', 'user_id_video_nui']
    feature = pd.merge(feature, act_feature, on='user_id', how='left')
    act_feature = act.groupby('user_id')['author_id'].nunique().reset_index()
    act_feature.columns = ['user_id', 'user_id_author_nui']
    feature = pd.merge(feature, act_feature, on='user_id', how='left')
    act_feature = act.groupby('user_id')['action_type'].nunique().reset_index()
    act_feature.columns = ['user_id', 'user_id_action_type_nui']
    feature = pd.merge(feature, act_feature, on='user_id', how='left')
    act_feature = act.groupby('user_id')['page'].nunique().reset_index()
    act_feature.columns = ['user_id', 'user_id_page_nui']
    feature = pd.merge(feature, act_feature, on='user_id', how='left')
    # unqiue/count
    feature['user_id_video_nui_ratio'] = feature['user_id_video_nui'] / feature['activity_day_count']
    feature['user_id_author_nui_ratio'] = feature['user_id_author_nui'] / feature['activity_day_count']
    feature['user_id_action_type_nui_ratio'] = feature['user_id_action_type_nui'] / feature['activity_day_count']
    feature['user_id_page_nui_ratio'] = feature['user_id_page_nui'] / feature['activity_day_count']
    # 行为时间差度量
    act_feature = act.groupby('user_id')['user_activity_day'].diff().fillna(0).reset_index(name='activity_diff')
    act['activity_diff'] = act_feature['activity_diff']
    result = act.groupby(['user_id'], as_index=False)['activity_diff'].agg({
        'activity_day_diff_max': 'max',
        'activity_day_diff_min': 'min',
        'activity_day_diff_mean': 'mean',
        # 'activity_day_diff_median': 'median',
        'activity_day_diff_ske': 'skew',
        # 'activity_day_diff_std': 'std',
        'activity_day_diff_last': 'last',
    })
    feature = pd.merge(feature, result, on='user_id', how='left')
    # 拍摄视频时间差峰度kurt
    # kurt = act.groupby(['user_id'])['activity_diff'].apply(pd.DataFrame.kurt).reset_index(name='activity_day_diff_kur')
    # feature = pd.merge(feature, kurt, on='user_id', how='left')
    # 统计总page 0 1 2 3 4 个数
    act_size = act.groupby(['user_id'])['page'].value_counts()
    act_size = pd.DataFrame(act_size).unstack()
    act_size.columns = ['page_0', 'page_1', 'page_2', 'page_3', 'page_4']
    # 统计总page 0 1 2 3 4 占总个数的频次
    act_sizeFill = act_size.fillna('0').astype('float')
    act_page_sum = act_sizeFill['page_0'] + act_sizeFill['page_1']+act_sizeFill['page_2']+act_sizeFill['page_3']+act_sizeFill['page_4']
    act_size['page0_percent'] = act_sizeFill['page_0']/act_page_sum
    act_size['page1_percent'] = act_sizeFill['page_1']/act_page_sum
    act_size['page2_percent'] = act_sizeFill['page_2']/act_page_sum
    act_size['page3_percent'] = act_sizeFill['page_3']/act_page_sum
    act_size['page4_percent'] = act_sizeFill['page_4']/act_page_sum
    act_size = act_size.reset_index()
    feature = pd.merge(feature, act_size, on='user_id', how='left')
    # 统计总action 0 1 2 3 4 5 个数
    act_size = act.groupby(['user_id'])['action_type'].value_counts()
    act_size = pd.DataFrame(act_size).unstack()
    act_size.columns = ['action_type_0', 'action_type_1', 'action_type_2', 'action_type_3', 'action_type_4', 'action_type_5']
    # 统计总action 0 1 2 3 4  5占总个数的频次
    act_sizeFill = act_size.fillna('0').astype('float')
    act_action_sum = act_sizeFill['action_type_0'] + act_sizeFill['action_type_1']+act_sizeFill['action_type_2']+act_sizeFill['action_type_3']+act_sizeFill['action_type_4']+act_sizeFill['action_type_5']
    act_size['action_type0_percent'] = act_sizeFill['action_type_0']/act_action_sum
    act_size['action_type1_percent'] = act_sizeFill['action_type_1']/act_action_sum
    act_size['action_type2_percent'] = act_sizeFill['action_type_2']/act_action_sum
    act_size['action_type3_percent'] = act_sizeFill['action_type_3']/act_action_sum
    act_size['action_type4_percent'] = act_sizeFill['action_type_4']/act_action_sum
    act_size['action_type5_percent'] = act_sizeFill['action_type_5']/act_action_sum
    act_size = act_size.reset_index()
    feature = pd.merge(feature, act_size, on='user_id', how='left')
    
    # 每天操作视频的数量特征
    act_day_sum = act.groupby(['user_id', 'user_activity_day']).size().rename('act_day_sum').reset_index()
    act_day_sum_var = act_day_sum.groupby('user_id')['act_day_sum'].var().rename('act_day_sum_var').reset_index()
    feature = feature.merge(act_day_sum_var, 'left', on='user_id')
    # rank的特征工程
    act['day_rank'] = act.groupby('user_id')['user_activity_day'].rank(ascending=False, method='dense')
    result = act.groupby(['user_id'], as_index=False)['day_rank'].agg({
        'rank_activity_day_count': 'count',
        'rank_activity_day_max': 'max',
        'rank_activity_day_min': 'min',
        'rank_activity_day_mean': 'mean',
        'rank_activity_day_median': 'median',
        'rank_activity_day_skew': 'skew',
        'rank_activity_day_std': 'std',
        'rank_activity_day_last': 'last',
    })
    feature = pd.merge(feature, result, on='user_id', how='left')
    
    # 查看每个用户观看相同page的特征
    result1 = act.groupby(['user_id', 'user_activity_day'], as_index=False)['page'].agg({
        'user_day_act_page_count': 'count'})
    result2 = result1.groupby(['user_id'], as_index=False)['user_day_act_page_count'].agg({
        'act_day_page_count_max': 'max',
        'act_day_page_count_min': 'min',
        'act_day_page_count_mean': 'mean',
        'act_day_page_count_std': 'std',
    })
    feature = pd.merge(feature, result2, on=['user_id'], how='left')
    del result1, result2
    # 查看每个用户观看相同action的特征
    result1 = act.groupby(['user_id', 'user_activity_day'], as_index=False)['action_type'].agg({
        'user_day_act_action_count': 'count'})
    result2 = result1.groupby(['user_id'], as_index=False)['user_day_act_action_count'].agg({
        'act_day_action_count_min': 'min',
        'act_day_action_count_std': 'std',
        'act_day_action_count_max': 'max',
        'act_day_action_count_mean': 'mean',
    })
    feature = pd.merge(feature, result2, on=['user_id'], how='left')
    
    # 添加时间连续特征
    act_max_lx_days = act.groupby('user_id').agg({'user_activity_day': get_max_lx}).reset_index().rename(columns={'user_activity_day': 'act_max_lx_days'})
    act_last_lx_days = act.groupby('user_id').agg({'user_activity_day': get_last_lx}).reset_index().rename(columns={'user_activity_day': 'act_last_lx_days'})
    act_lx_var = act.groupby('user_id').agg({'user_activity_day': get_lx_var}).reset_index().rename(columns={'user_activity_day': 'act_lx_var'})
    feature = pd.merge(feature, act_max_lx_days, on='user_id', how='left')
    feature = pd.merge(feature, act_last_lx_days, on='user_id', how='left')
    feature = pd.merge(feature, act_lx_var, on='user_id', how='left')
    
    print('activity表特征提取完毕',int(time.time()-s),'s')
    print('changdu', len(feature['user_id']))

    # 距离最后天数的特征集
    act1 = act[act['user_activity_day'] == (ls_date-1)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_1_activity_cout'})
    act2 = act[act['user_activity_day'] == (ls_date-2)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_2_activity_cout'})
    act3 = act[act['user_activity_day'] == (ls_date-3)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_3_activity_cout'})
    act4 = act[act['user_activity_day'] == (ls_date-4)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_4_activity_cout'})
    act5 = act[act['user_activity_day'] == (ls_date-5)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_5_activity_cout'})
    act6 = act[act['user_activity_day'] == (ls_date-6)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_6_activity_cout'})
    feature = pd.merge(feature, act1, on='user_id', how='left')
    feature = pd.merge(feature, act2, on='user_id', how='left')
    feature = pd.merge(feature, act3, on='user_id', how='left')
    feature = pd.merge(feature, act4, on='user_id', how='left')
    feature = pd.merge(feature, act5, on='user_id', how='left')
    feature = pd.merge(feature, act6, on='user_id', how='left')
    
    # 添加时间划窗的特征3,5,7天
    act_day_3 = act[(act['user_activity_day']<=ls_date) & (act['user_activity_day']>ls_date-3)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_day3_activity_cout'})
    act_day_3['ls_day3_activity_cout'] = 0.7*act_day_3['ls_day3_activity_cout']
    feature = pd.merge(feature, act_day_3, on='user_id', how='left')
    act_day_5 = act[(act['user_activity_day']<=ls_date) & (act['user_activity_day']>ls_date-5)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_day5_activity_cout'})
    act_day_5['ls_day5_activity_cout'] = 0.3*act_day_5['ls_day5_activity_cout']
    feature = pd.merge(feature, act_day_5, on='user_id', how='left')
    act_day_7 = act[(act['user_activity_day']<=ls_date) & (act['user_activity_day']>ls_date-7)].groupby(['user_id']).size().reset_index().rename(columns={0: 'ls_day7_activity_cout'})
    act_day_7['ls_day7_activity_cout'] = 0.1*act_day_7['ls_day7_activity_cout']
    feature = pd.merge(feature, act_day_7, on='user_id', how='left')
    
    # 附加特征
    feature['register_days'] = ls_date - feature['register_day']
    
    return feature


def get_data_feature():
    first_train_data = get_train_label(first_train_path, first_test_path, 16)
    first_feature = deal_feature(first_train_path, first_train_data['user_id'], 16)
    first_feature['label'] = first_train_data['label']
    print('第一组训练数据特征提取完毕')

    sec_train_data = get_train_label(sec_train_path, sec_test_path, 23)
    sec_feature = deal_feature(sec_train_path, sec_train_data['user_id'], 23)
    sec_feature['label'] = sec_train_data['label']
    print('第二组训练数据特征提取完毕')

    first_feature.to_csv(train_path, index=False)
    sec_feature.to_csv(val_path, index=False)
    print('训练数据存储完毕')

    test_data = get_test(thr_dataSet_path)
    test_feature = deal_feature(thr_dataSet_path, test_data['user_id'], 30)
    test_feature.to_csv(test_path, index=False)
    print('测试数据存储完毕')

get_data_feature()

import lightgbm as lgb
import sklearn.metrics
print('开始处理特征......')
train_path = './train.csv'
val_path = './val.csv'
test_path = './test.csv'

train = pd.read_csv(train_path)
val = pd.read_csv(val_path)
test = pd.read_csv(test_path)

used_feature = [
    
    # 基础特征
    'register_type', 'device_type', # 'register_day',

    # create 特征
    'create_day_count', 'create_day_max', 'create_day_min', 'create_day_mean', 'create_day_last',
    'create_day_diff_mean', 'create_day_diff_min', 'create_day_diff_max', 'create_day_diff_ske', 'create_day_diff_last',
    'create_register_day', 'create_day_uni', 'create_day_radio',
    'create_register_day_min', 'create_register_day_mean',
    # register 特征
    'register_day_count', 'register_register_day', # 'register_day_mean',
    # 'register_day_last',  'register_day_uni', 'register_day_radio', 'register_register_day',

    # launch 特征
    'launch_day_count',
    'launch_day_diff_max', 'launch_day_diff_min', 'launch_day_diff_mean', 'launch_day_diff_ske', 'launch_day_diff_last',
    'launch_register_day',
    'launch_day_mean', 'launch_day_max',  'launch_day_last', 'launch_day_min',
    'launch_day_uni', 'launch_day_radio', 'activity_register_day_first',
    # activity 特征
    'activity_day_count',
    'user_id_page_nui', 'user_id_page_nui_ratio', 'user_id_video_nui', 'user_id_video_nui_ratio',
    'user_id_author_nui', 'user_id_author_nui_ratio', # 'user_id_action_type_nui', 'user_id_action_type_nui_ratio',
    'activity_register_day', 'activity_day_max', 'activity_day_min', 'activity_day_mean', 'activity_day_last',
    'activity_day_diff_max', 'activity_day_diff_min', 'activity_day_diff_mean', 'activity_day_diff_ske',
    'activity_day_diff_last',
    'page_0', 'page_1', 'page_2', 'page_3', 'page_4', 'page0_percent', 'page1_percent', 'page2_percent', 'page3_percent', 'page4_percent',
    'action_type_0', 'action_type_1', 'action_type_2', 'action_type_3', 'action_type_4', 'action_type_5',
    # 'action_type0_percent', 'action_type1_percent', 'action_type2_percent', 'action_type3_percent', 'action_type4_percent', 'action_type5_percent',
    'activity_register_day_min',
    
    'ls_1_launch_cout', 'ls_2_launch_cout', 'ls_3_launch_cout', # 'ls_4_launch_cout', 'ls_5_launch_cout', 'ls_6_launch_cout',
    'ls_1_create_cout', 'ls_2_create_cout', 'ls_3_create_cout', # 'ls_4_create_cout', 'ls_5_create_cout', 'ls_6_create_cout',
    'ls_1_activity_cout', 'ls_2_activity_cout', 'ls_3_activity_cout', # 'ls_4_activity_cout', 'ls_5_activity_cout', 'ls_6_activity_cout',
    'act_count_sum', 'last_login_time', 'last_act_time', 'last_video_time', 'register_days',
    'login_count_sum', 'video_count_sum',
    
    'act_day_sum_var', 'lau_day_sum_var', 'video_day_sum_var',
    
    # 'rank_create_day_count', 'rank_create_day_max', 'rank_create_day_min', 'rank_create_day_mean', 'rank_create_day_median', 'rank_create_day_skew', 'rank_create_day_std', # 'rank_create_day_last'
    # 'rank_launch_day_last', 'rank_launch_day_median', 'rank_launch_day_skew', 'rank_launch_day_std', # 'rank_launch_day_max', 'rank_launch_day_count', 'rank_launch_day_min', 'rank_launch_day_mean'
    # 'rank_activity_day_count', 'rank_activity_day_max', 'rank_activity_day_min', 'rank_activity_day_mean', 'rank_activity_day_last', 'rank_activity_day_median', 'rank_activity_day_skew', 'rank_activity_day_std',
   
    'act_day_page_count_max', 'act_day_page_count_min', 'act_day_page_count_mean', 'act_day_page_count_std',
    'act_day_action_count_min', 'act_day_action_count_std', 'act_day_action_count_max', 'act_day_action_count_mean',
    # 稍微提升
    'ls_day3_create_cout', 'ls_day5_create_cout', 'ls_day7_create_cout',
    'ls_1_greister_cout', 'ls_2_greister_cout', 'ls_3_greister_cout',
    'ls_day3_register_cout', 'ls_day5_register_cout', 'ls_day7_register_cout',
    'ls_day3_launch_cout', 'ls_day5_launch_cout', 'ls_day7_launch_cout',
    'ls_day3_activity_cout', 'ls_day5_activity_cout', 'ls_day7_activity_cout',
    # 连续时间的特征
    'cre_max_lx_days', 'cre_last_lx_days', 'cre_lx_var',
    'reg_max_lx_days', 'reg_last_lx_days', 'reg_lx_var',
    'lau_max_lx_days', 'lau_last_lx_days', 'lau_lx_var',
    'act_max_lx_days', 'act_last_lx_days', 'act_lx_var',
]
used_feature = np.array(used_feature)
print(used_feature)
train_feature = train[used_feature]
val_feature = val[used_feature]
test_feature = test[used_feature]
train_label = train['label']
val_label = val['label']

print('特征处理完毕.....')

###################### lgb ##########################
import lightgbm as lgb

print('载入数据......')
lgb_train = lgb.Dataset(train_feature, train_label)
lgb_eval = lgb.Dataset(val_feature, val_label, reference=lgb_train)


print('开始训练......')

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'}
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=57,
                valid_sets=lgb_eval,
                early_stopping_rounds=20
                )

temp = gbm.predict(val_feature)
print('AUC结果：' + str(sklearn.metrics.roc_auc_score(val_label, temp)))
print('log_loss结果：' + str(sklearn.metrics.log_loss(val_label, temp)))
# 查看属性重要性
df = pd.DataFrame(columns=['feature', 'important'])
df['feature'] = used_feature
df['important'] = gbm.feature_importance()
df = df.sort_values(axis=0, ascending=True, by='important')
print (df)

########################## 结果 ############################
train = pd.concat([train_feature, val_feature])
label = pd.concat([train_label, val_label])
lgb_train = lgb.Dataset(train, label)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=57,
                )
pre = gbm.predict(test_feature)
df_result = pd.DataFrame()
df_result['user_id'] = test['user_id']
df_result['result'] = pre
df_result.to_csv('./lgb_resul.txt', index=False, header=None)