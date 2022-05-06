import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math as mt
# Transformer
from sklearn.preprocessing import PowerTransformer
# Regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
# Scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# Encoder
from sklearn.preprocessing import OneHotEncoder
# Split
from sklearn.model_selection import train_test_split
# Balance Data
from imblearn.over_sampling import SMOTE
# Error Matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
# misc
import pickle

def read_df (filepath):
    import pandas as pd
    
    df = pd.read_csv(filepath)
    pd.set_option('display.max_columns', None)
    df = df.drop(['comments'], axis =1)
    df = df.drop(['state'], axis =1)
    
    return df

def standardize_cols (df):
    cols = []
    for column in df.columns:
        cols.append(column.lower().replace(' ','_'))#fill in the list with all column names in lowercase
    df.columns = cols#replace the dataframe columns with the columns stored in the list
    
    return df

def check_na (df):
    for i in df.columns:
        print(i, df[i].value_counts(dropna = False))
      
def clean_countries (x, country_list):
    if x in country_list:
        return 'Other'
    else:
        return x

def num_cont_disc (df, num_uniques):
    
    df_disc = df.loc[:,df.nunique() < num_uniques] #create dataframes based on filtered nuniques
    df_cont = df.loc[:,df.nunique() > num_uniques] #create dataframes based on filtered nuniques

    return df_disc, df_cont

    
def fill_na (df): 
    df = df.fillna('Unknown')
    
    return df

def clean_age (df, col):
    import numpy as np
    import pandas as pd
    import math
    
    remove = [99999999999,-1726,-29, -1,5,8,11,329]
    filtered_mean = df[df[col].isin(remove) == False][col].mean()
    df[col] = np.where(df[col].isin(remove), filtered_mean, df[col])
    
    trunc = lambda x: math.trunc(1000000 * x) / 1000000;
    df[[col]].applymap(trunc)
    df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df[col]


def onehot_categories (df, df_cat_nom):
    categories = [ list(df[col].unique()) for col in df_cat_nom.columns ]
    
    return categories

def outliers_out(df, columns):
 
    df2 = df.copy()
    i = 0

    for col in columns:
        iqr = np.percentile(df2[columns[i]],75) - np.percentile(df2[columns[i]],25)
        upper_limit = np.percentile(df2[columns[i]],75) + 1.5*iqr
        lower_limit = np.percentile(df2[columns[i]],25) - 1.5*iqr
        df2 = df2[(df2[columns[i]]>lower_limit) & (df2[columns[i]]<upper_limit)]
        sns.displot(df2[columns[i]])
        
        i+=1
        
    plt.show()
    return df2

def gender_standardize(x):
    
    females = ['Female','female','Woman','woman','Femake','Femail','femme','F']
    males = ['M','Male','Maile','male','Mal','Make','Man','Msle','Mail','Malr','man']
    found_female = False
    found_male = False
    for elem in females:
        if ( elem in x ):
            return "Female"
            found_female = True
            break
    if ( found_female == False ):
        for elem in males:
            if ( elem in x ):
                return "Male"
                found_male = True
                break
    if ( (found_female == False) and (found_male == False) ):
        return "Other"
        

def log_transform_clean(x):
    
    if np.isfinite(x) and x!=0:
        return np.log(x)
    else:
        return np.NAN

def log_it(df):
    
    data_log = pd.DataFrame()
    for item in df.columns:
        data_log[item] = df[item].apply(log_transform_clean)
        
    return data_log

def df_box_cox(df):
    df1 = pd.DataFrame()
    for item in df.columns:
        if df[item].min() > 0:
            df1[item] = df[item]

    return df1

def power_transform(df):
    
    df_f_bc = df_box_cox(df)
    pt_bc = PowerTransformer(method="box-cox")
    pt_bc.fit(df_f_bc)
    df_bc = pd.DataFrame(pt_bc.transform(df_f_bc), columns = df_f_bc.columns)
    
    pt_yj = PowerTransformer()
    pt_yj.fit(df)
    df_yj = pd.DataFrame(pt_yj.transform(df), columns = df.columns)
    
    return df_bc, df_yj

def plot_transformer(df):

    df = df.select_dtypes(np.number)
    data_log = pd.DataFrame()
    data_log = log_it(df)
    data_bc, data_yj = power_transform(df)
    r = df.shape[1]
    c = 4
    fig, ax = plt.subplots(r, c, figsize=(30,30))
    i = 0
    data = ""
    for item in df.columns:
        for j in range(c):
            if j == 0:
                data = df
                head = "original"
            elif j == 1:
                data = data_log
                head = "log"
            elif j == 2:
                data = data_yj
                head = "yeo-johnson"
            elif j == 3:
                data = data_bc
                head = "box-cox"
            ax[0,j].set_title(head)
         
            if item in data.columns:
                sns.histplot(a = data, x = item, ax = ax[i, j]); 
        i = i + 1

    plt.show()
    
def powertrans_plot_single(df, column):
    
    df2 = df.copy()
    df2 = df2[[column]]
    transformer = PowerTransformer().fit(np.array(df2[column]).reshape(-1,1))
    transf = transformer.transform(np.array(df2[column]).reshape(-1,1))
    df2_transf = pd.DataFrame(transf, columns = df2.columns)
    sns.displot(df2_transf[column])
    plt.show()

def powertrans_plot_multi(df):

    df2 = df.copy()
    df2 = df2.select_dtypes(include = np.number)
    transformer = PowerTransformer()
    transformer.fit(df2)
    ptrans_df = pd.DataFrame(transformer.transform(df2), columns = df2.columns)

    for column in ptrans_df.columns:   
        sns.displot(ptrans_df[column]);
        plt.show()

def plot_num (df):
    cols =[]
    df2 = df.select_dtypes(include = np.number)
    for column in df2.columns:
        cols.append(column)   
        sns.displot(df2[cols]);
        plt.show()
        cols.pop()

def boxplotting (df):
    cols = []
    df2 = df.select_dtypes(include = np.number)
    for column in df2.columns:
        cols.append(column)
    for col in cols:
        sns.boxplot(x = df2[col])
        plt.show()
        

def countplot(df):

    df2 = df.select_dtypes(include = object)
    for column in df2.columns:
        plt.subplots(figsize=(12,4))
        sns.countplot(x= column, data= df2, palette="Set2");
        plt.show()
    

def plot_transformer(df):

    df = df.select_dtypes(np.number)
    data_log = pd.DataFrame()
    data_log = log_it(df)
    data_bc, data_yj = power_transform(df)
    r = df.shape[1]
    c = 4
    fig, ax = plt.subplots(r, c, figsize=(30,30))
    i = 0
    data = ""
    for item in df.columns:
        for j in range(c):
            if j == 0:
                data = df
                head = "original"
            elif j == 1:
                data = data_log
                head = "log"
            elif j == 2:
                data = data_yj
                head = "yeo-johnson"
            elif j == 3:
                data = data_bc
                head = "box-cox"
            ax[0,j].set_title(head)
         
            if item in data.columns:
                sns.histplot(a = data[item], ax = ax[i, j]) 
        i = i + 1
    plt.show()

def train_models(list_of_models, X_train, X_test, y_train, y_test):
    
    '''
    This function does the following things:
    1. Trains and returns models mentioned in a list
    2. Returns a list of arrays with the predictions of each model on for both Train and Test set
    3. Returns a list with R2 Scores for both Train and Test set
    4. Stores the models in .pkl files
    '''
    
    import pickle
  
    train_preds = []
    test_preds  = []
    train_r2_scores = []
    test_r2_scores = []
        
    for model in list_of_models:
        
        model[0].fit(X_train, y_train)
        filename = model[1] + ".pkl"
        
        with open(filename,"wb") as file:
            pickle.dump(model[0],file)
            
        
        y_pred_train = model[0].predict(X_train)
        y_pred_test  = model[0].predict(X_test) 
        
        train_preds.append( y_pred_train )
        test_preds.append( y_pred_test )
        
        train_r2_scores.append( r2_score(y_train, y_pred_train ) )
        test_r2_scores.append( r2_score(y_test, y_pred_test ) )
                                                              
    
    return list_of_models, train_preds, test_preds, train_r2_scores, test_r2_scores
    
def conf_matrix (y_train, y_train_pred, y_test, y_test_pred, model_name):
   
    cm_train = confusion_matrix(y_train, y_train_pred)
    disp = ConfusionMatrixDisplay(cm_train,display_labels=model_name.classes_);
    disp.plot()
    plt.title('Confusion Matrix for Train Set')
    plt.show()
    
    cm_test = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(cm_test,display_labels=model_name.classes_);
    disp.plot()
    plt.title('Confusion Matrix for Test Set')
    plt.show()

    return

def log_class_scores(y_train, y_train_pred, y_test, y_test_pred, label):
    print('ACCURACY SCORE:\n')
    print("The accuracy in the TRAIN set is: {:.3f}".format(accuracy_score(y_train, y_train_pred)))
    print("The accuracy in the TEST set is: {:.3f}".format(accuracy_score(y_test, y_test_pred)))
    print('\n-------------------------------------------------\n')
    print('PRECISSION SCORE:\n')
    print("The precission in the TRAIN set is: {:.3f}".format(precision_score(y_train, y_train_pred, pos_label=label)))
    print("The precission in the TESTset is: {:.3f}".format(precision_score(y_test, y_test_pred, pos_label=label)))
    print('\n-------------------------------------------------\n')
    print('RECALL SCORE:\n')
    print("The recall in the TRAIN set is: {:.3f}".format(recall_score(y_train, y_train_pred, pos_label=label)))
    print("The recall in the TEST set is: {:.3f}".format(recall_score(y_test,  y_test_pred, pos_label=label)))
    print('\n-------------------------------------------------\n')
    print('F1 SCORE:\n')
    print("The F1-score for the TRAIN set is {:.2f}".format(f1_score(y_train,y_train_pred, pos_label=label)))
    print("The F1-score for the TEST set is {:.2f}".format(f1_score(y_test,y_test_pred, pos_label=label)))
    print('\n-------------------------------------------------\n')
    print('KAPPA SCORE:\n')
    print("The Kappa in the TRAIN set is: {:.2f}".format(cohen_kappa_score(y_train,y_train_pred)))
    print("The Kappa in the TEST set is: {:.2f}".format(cohen_kappa_score(y_test,y_test_pred)))
    print('\n-------------------------------------------------\n')
    print('CLASSIFICATION REPORT FOR TRAIN SET:\n')
    print(classification_report(y_train, y_train_pred,target_names=['No','Yes']))
    print('\n-------------------------------------------------\n')
    print('CLASSIFICATION REPORT FOR TEST SET:\n')
    print(classification_report(y_test, y_test_pred,target_names=['No','Yes']))
    
    return

def var_rank (model_name, transformed_df, trainortest):
    import numpy as np
    rank = [(value, index) for index, value in enumerate(np.abs(model_name.coef_[0]).tolist())]
    rank.sort(reverse = True)
    rank = [ (transformed_df.columns[elem[1]], round(elem[0],2)) for elem in rank ]
    print('Ranking in descending order of the variables that affected the predictions the most for the', trainortest, 'set are:')
    print('------------------------------------------------------------------------------------------')
    
    return rank


def train_models(list_of_models, X_train, X_test, y_train, y_test):
    
    '''
    This function does the following things:
    1. Trains and returns models mentioned in a list
    2. Returns a list of arrays with the predictions of each model on for both Train and Test set
    3. Returns a list with R2 Scores for both Train and Test set
    4. Stores the models in .pkl files
    '''
    
    import pickle
  
    train_preds = []
    test_preds  = []
    train_r2_scores = []
    test_r2_scores = []
        
    for model in list_of_models:
        
        model[0].fit(X_train, y_train.values.ravel())
        filename = model[1] + ".pkl"
        
        with open(filename,"wb") as file:
            pickle.dump(model[0],file)
            
        
        y_pred_train = model[0].predict(X_train)
        y_pred_test  = model[0].predict(X_test) 
        
        train_preds.append( y_pred_train )
        test_preds.append( y_pred_test )
        
        train_r2_scores.append( r2_score(y_train.values.ravel(), y_pred_train ) )
        test_r2_scores.append( r2_score(y_test.values.ravel(), y_pred_test ) )
                                                              
    
    return list_of_models, train_preds, test_preds, train_r2_scores, test_r2_scores


def reg_performance (y_train, y_pred_train, y_test, y_pred_test):
    
    '''
    Measures the performance of a single Regression Model.
    '''

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    ME_train = round(np.mean(y_train.values - y_pred_train),2)
    ME_test  = round(np.mean(y_test.values - y_pred_test),2)

    MAE_train = round(mean_absolute_error(y_train.values,y_pred_train),2)
    MAE_test  = round(mean_absolute_error(y_test.values,y_pred_test),2)

    MSE_train = round(mean_squared_error(y_train.values,y_pred_train),2)
    MSE_test  = round(mean_squared_error(y_test.values,y_pred_test),2)

    RMSE_train = round(np.sqrt(MSE_train),2)
    RMSE_test  = round(np.sqrt(MSE_test),2)

    MAPE_train = round(np.mean((np.abs(y_train.values-y_pred_train) / y_train.values)* 100.),2)
    MAPE_test  = round(np.mean((np.abs(y_test.values-y_pred_test) / y_test.values)* 100.),2)

    R2_train = round(r2_score(y_train.values,y_pred_train),2)
    R2_test  = round(r2_score(y_test.values,y_pred_test),2)
    
    
    print('PERFORMANCE METRICS')
    print('--------------------')

    performance = pd.DataFrame({'Error_metric': ['Mean error','Mean absolute error','Mean squared error',
                                             'Root mean squared error','Mean absolute percentual error',
                                             'R2'],
                            'Train': [ME_train, MAE_train, MSE_train, RMSE_train, MAPE_train, R2_train],
                            'Test' : [ME_test, MAE_test , MSE_test, RMSE_test, MAPE_test, R2_test]})

    display(performance)
    
    print('REAL vs PREDICTED PERFORMANCE')
    print('------------------------------')
    
    #Creating a DataFrame to show differences between predicted and Real values on Train Set:
    df_train = pd.DataFrame()
    df_train['Real_train'] = y_train
    df_train['Pred_train'] = y_pred_train

    #Creating a DataFrame differences between predicted and Real values on Test Set:
    df_test = pd.DataFrame()
    df_test['Real_test'] = y_test
    df_test['Pred_test'] = y_pred_test

    display(df_train.head())
    display(df_test.head())
    
    return performance, df_train, df_test

def lr_perf_plots(df_train, df_test):

    '''
    Provides a scatter plot combined with a lineplot to visually asess
    the performance of your model

    '''
    
    fig2, ax2 = plt.subplots(2,2, figsize=(16,8))

    sns.scatterplot(y = df_train['Pred_train'], x=df_train['Real_train'], ax = ax2[0,0])
    sns.lineplot(data = df_train, x = 'Real_train', y = 'Real_train', color = 'black', ax = ax2[0,0])
    sns.histplot(df_train['Real_train'] - df_train['Pred_train'], ax = ax2[0,1])

    sns.scatterplot(y = df_test['Pred_test'], x=df_test['Real_test'], ax = ax2[1,0])
    sns.lineplot(data = df_test, x = 'Real_test', y = 'Real_test', color = 'black', ax = ax2[1,0])
    sns.histplot(df_test['Real_test'] - df_test['Pred_test'], ax = ax2[1,1])
    
    plt.show()


def reg_score_comparison(train_r2_scores, test_r2_scores, model_list):
    
    for i in range(0, len(train_r2_scores)):
        
        R2_train = train_r2_scores[i]
        model_train = model_list[i][1]
        
        print('The R2 score of the', model_train, ' on Train set is: ', round(R2_train,2))
        
        i +=1
    
    print('---------------------------------------------------------')
    
    for j in range(0, len(test_r2_scores)):
        
        R2_test = test_r2_scores[j]
        model_test = model_list[j][1]
        
        print('The R2 score of the', model_test, 'on Test set is: ', round(R2_test,2))
        
        j +=1

def binning (df, col, num_groups, label_names = None):
    '''
    Documentation:
    This function replace the number of possible values inside a dataframe column, to have tthe same amount of
    rows for each group.
    
    Input values: 
    df -> Pandas dataframe
    col -> The column of the dataframe to do the transformation
    num_groups -> Number of desired groups for the dataframe column 'col'
    label_names (optional)-> Labels of each group. If it's not provided, automated group labels will be generated.
    
    Output:
    The colum values with the new groups.
    '''
    df2 = df.copy()
    if (df2[col].nunique() < num_groups):
        return df2[col]
    else:
        labels = []
        if (label_names == None):
            for i in range(num_groups):
                labels.append('Group_'+ str(i+1))
            df2[col] = pd.qcut(df2[col], num_groups, labels=labels)
            return df2[col]
        else:
            df2[col] = pd.qcut(df2[col], num_groups, labels=label_names)
            return df2[col]
    



    


    


