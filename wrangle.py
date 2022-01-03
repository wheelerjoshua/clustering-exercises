import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import env


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import os
from env import host, user, password

####### Acquire #######
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
def new_zillow_data():
    '''
    This function reads all tables and columns into a dataframe, including only properties with lat/long
    data and had transactions in 2017. Properties with multiple transactions 
    will display only the most recent transaction.
    '''
    sql_query = """
                SELECT p.*, m.logerror, m.transactiondate, ac.airconditioningdesc, arch.architecturalstyledesc, b.buildingclassdesc, heat.heatingorsystemdesc, pt.propertylandusedesc, s.storydesc, c.typeconstructiondesc
                FROM properties_2017 as p
                JOIN predictions_2017 as m USING(parcelid)
                LEFT JOIN airconditioningtype as ac USING(airconditioningtypeid)
                LEFT JOIN architecturalstyletype as arch USING(architecturalstyletypeid)
                LEFT JOIN buildingclasstype as b USING(buildingclasstypeid)
                LEFT JOIN heatingorsystemtype as heat USING(heatingorsystemtypeid)
                LEFT JOIN propertylandusetype as pt USING(propertylandusetypeid)
                LEFT JOIN storytype as s USING(storytypeid)
                LEFT JOIN typeconstructiontype as c USING(typeconstructiontypeid)
                LEFT JOIN unique_properties as u USING(parcelid)
                INNER JOIN (SELECT p.parcelid, MAX(transactiondate) AS maxdate FROM properties_2017 as p JOIN predictions_2017 USING(parcelid) GROUP BY p.parcelid, logerror) md ON p.parcelid = md.parcelid AND transactiondate = maxdate
                WHERE transactiondate LIKE '2017%%' AND latitude IS NOT NULL AND longitude IS NOT NULL
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df

def get_zillow_data():
    '''
    This function reads in zillow data from the Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
        
        # Cache data
        df.to_csv('zillow.csv')
        
    return df




###### Prepare #######
def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def drop_cols(df, cols_to_drop):
    df.drop(columns = cols_to_drop, inplace = True)
    return df

def wrangle_zillow():
    # acquire df
    df = get_zillow_data()
    # only single family
    df = df[df.propertylandusetypeid == 261]
    # at least 1 bed and bath and 350 sqft
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & (df.calculatedfinishedsquarefeet>350)]
    # handle missing values
    df = handle_missing_values(df)
    # drop unnecessary columns
    df = drop_cols(df, ['id','calculatedbathnbr', 'buildingqualitytypeid','finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid','heatingorsystemdesc','propertycountylandusecode', 'propertylandusetypeid','propertyzoningdesc',  'censustractandblock', 'propertylandusedesc', 'unitcnt'])
    # fill lotsize
    df.lotsizesquarefeet.fillna(7313, inplace = True)
    # properties under 5 million USD
    df = df[df.taxvaluedollarcnt < 5_000_000]
    # add counties
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',np.where(df.fips == 6059, 'Orange', 'Ventura'))  
    # catch other nulls
    df.dropna(inplace=True)
    # return wrangled df
    return df
 
    
####### Prepare ########    

def min_max_scaler(train, valid, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    num_vars = list(train.select_dtypes('number').columns)
    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    train[num_vars] = scaler.fit_transform(train[num_vars])
    valid[num_vars] = scaler.transform(valid[num_vars])
    test[num_vars] = scaler.transform(test[num_vars])
    return scaler, train, valid, test

def remove_outliers(df, k, col_list):
    ''' 
    Takes in a df, k, and list of columns returns
    a df with removed outliers
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

######## Split ########
def split_data(df):
    '''
    Takes in a dataframe and returns train, validate, and test subset dataframes. 
    '''
    train, test = train_test_split(df, test_size = .2, random_state = 222)
    train, validate = train_test_split(train, test_size = .3, random_state = 222)
    return train, validate, test




#######

def get_mall_customers(sql):
	    url = get_connection('mall_customers')
	    mall_df = pd.read_sql(sql, url, index_col='customer_id')
	    return mall_df

def wrangle_mall_df():
    
    # acquire data
    sql = 'select * from customers'


    # acquire data from SQL server
    mall_df = get_mall_customers(sql)
    
    # handle outliers
    mall_df = remove_outliers(mall_df, 1.5, ['age', 'spending_score', 'annual_income'])
    
    # get dummy for gender column
    dummy_df = pd.get_dummies(mall_df.gender, drop_first=True)
    mall_df = pd.concat([mall_df, dummy_df], axis=1).drop(columns = ['gender'])
    mall_df.rename(columns= {'Male': 'is_male'}, inplace = True)
    # return mall_df

    # split the data in train, validate and test
    train, test = train_test_split(mall_df, train_size = 0.8, random_state = 123)
    train, validate = train_test_split(train, train_size = 0.75, random_state = 123)
    
#     return min_max_scaler, train, validate, test
    return mall_df