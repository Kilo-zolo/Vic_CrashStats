import sys
sys.path.append('/path/to/directory/containing/data_prep')
from data_prep import load_and_prep_data
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def feature_eng(loaded_data):
    print("Passing data...")
    accident_data = loaded_data['accident_data']
    person_data = loaded_data['person_data']
    atmospheric_data = loaded_data['atmospheric_data']
    road_data = loaded_data['road_data']
    event_data = loaded_data['event_data']
    node_complex_data = loaded_data['node_complex_data']

    print("Setting index...")
    df = accident_data.set_index('ACCIDENT_NO')

    ## datetime columns need to be treated as numeric
    print("Treating datetime as numeric...")
    df['YEAR'] = df['ACCIDENTDATE'].dt.year
    df['MONTH'] = df['ACCIDENTDATE'].dt.month
    df['HOUR'] = df['ACCIDENTTIME'].dt.hour

    ## convert age column to numeric data
    person_data['AGE'] = pd.to_numeric(person_data['AGE'], errors='coerce')

    ## mean age of all involved and mean age of drivers
    print("Creating MEAN_AGE & MEAN_DRIVER_AGE columns...")
    df['MEAN_AGE'] = person_data.groupby('ACCIDENT_NO')['AGE'].mean()
    df['MEAN_DRIVER_AGE'] = person_data[person_data['Road User Type Desc'] == 'Drivers'].groupby('ACCIDENT_NO')['AGE'].mean()

    ## how many drivers of each sex
    print("Creating FEMALE_DRIVERS & MALE_DRIVERS columns...")
    df[['FEMALE_DRIVERS', 'MALE_DRIVERS']] = person_data[person_data['Road User Type Desc']=='Drivers'].groupby('ACCIDENT_NO')['SEX'].value_counts().unstack().fillna(0).drop(['U', ' '], axis=1)

    ## if our newly newly created numeric columns have nulls i.e no drivers in the accident
    ## fill numeric nulls with -1
    print("Filling all null numeric value with -1...")
    cols = ['MEAN_AGE', 'MEAN_DRIVER_AGE', 'FEMALE_DRIVERS', 'MALE_DRIVERS']
    df[cols] = df[cols].fillna(-1)

    ## MultiLabelBinarizing atmospheric conditions
    print("MultilabelBinarizing atmospheric conditions...")
    atmospheric_df_to_join = atmospheric_data.groupby('ACCIDENT_NO')['Atmosph Cond Desc'].value_counts().unstack().fillna(0)
    print("Joining atmospheric dframe to main dframe...")
    df = pd.concat([df, atmospheric_df_to_join], axis=1)

    ## MultiLabelBinarizing road conditions
    print("MultilabelBinarizing road conditions...")
    road_df_to_join = road_data.groupby('ACCIDENT_NO')['Surface Cond Desc'].value_counts().unstack().fillna(0)
    print("Joining road conditions dframe to main dframe...")
    df = pd.concat([df, road_df_to_join], axis=1)

    ## MultiLabelBinarizing event type
    print("MultilabelBinarizing event type...")
    event_df_to_join = event_data.groupby('ACCIDENT_NO')['Event Type Desc'].value_counts().unstack().fillna(0)
    print("Joining events dframe to main dframe...")
    df = pd.concat([df, event_df_to_join], axis=1)

    ## was there a complex intersection in the accident
    print("Figuring out if there was any complex intersections...")
    df['COMPLEX_INT'] = node_complex_data.groupby('ACCIDENT_NO')['COMPLEX_INT_NO'].any()

    ## getting rid of 999s on speed_zone
    print("Removing all 999 speed zones...")
    df['SPEED_ZONE'] = df['SPEED_ZONE'].replace(999, df['SPEED_ZONE'].mode()[0])

    ## binning low levels on DCA description
    print("Binning all rare values into Other")
    def bin_low_levels(s, thresh):
        vc = s.value_counts(normalize=True)
        return s.apply(lambda x: 'Other' if x not in vc[vc>thresh].index else x)

    df['DCA Description'] = bin_low_levels(df['DCA Description'], 0.01)

    ## dropping the datetime columns and the rarely seen level 4 severity (no injured persons)
    df.drop(['ACCIDENTDATE', 'ACCIDENTTIME', 'NODE_ID'], axis=1, inplace=True)
    df = df[df['SEVERITY']!=4]
    print("Successfully created usable dataframe...")
    return df

def pre_process(main_df):
    ## numeric fields
    print("Allocating numeric columns...")
    num_cols = ['NO_OF_PERSONS', 'NO_OF_VEHICLES', 'SPEED_ZONE', 'MEAN_AGE', 'MEAN_DRIVER_AGE', 'NO_OF_FEMALE_DRIVERS', 
                'NO_OF_MALE_DRIVERS', 'YEAR', 'MONTH', 'HOUR']

    ## categorical fields
    print("Allocating Categorical columns...")
    cat_cols = ['Accident Type Desc', 'Day Week Description', 'DCA Description', 'Light Condition Desc', 'Road Geometry Desc',
                'COMPLEX_INT']
    
    ## Converting SEVERITY to a numerical column
    main_df['SEVERITY'] = pd.to_numeric(main_df['SEVERITY'], errors='coerce')

    ## setting our dependent variable 'y' as the SEVERITY col
    print("Setting up SEVERITY as dependant variable...")
    y = main_df['SEVERITY']

    ## the values of y start at 1 but for the ordinal model to work best we should start them at 0
    print("Decreasing SEVERITY identifiers by 1...")
    y = y.apply(lambda x: x-1)

    ## dropping SEVERITY from our independent variable
    print("Dropping SEVERITY from our features...")
    main_df.drop('SEVERITY', axis=1, inplace=True)

    ## split data into train and test
    print("Splitting date into train and test sets...")
    x_train, x_test, y_train, y_test= train_test_split(main_df, y, test_size=0.2)

    ## creating binary dependent variable
    print("Creating binary dependant variables...")
    y_train_binary, y_test_binary = y_train.apply(lambda x: x<2), y_test.apply(lambda x: x<2)

    ## one-hot encoding
    print("One hot encoding the data...")
    hot_encode = OneHotEncoder(handle_unknown='ignore', sparse=False)
    x_train_hot_encode = x_train[cat_cols].to_numpy()
    x_test_hot_encode = x_test[cat_cols].to_numpy()
    hot_encode.fit(x_train_hot_encode)

    x_train_hot_encode = hot_encode.transform(x_train_hot_encode)
    print("Converting hot-encoded x_train to dataframe...")
    x_train_hot_encode = pd.DataFrame(x_train_hot_encode, index=x_train.index, columns=hot_encode.get_feature_names_out())

    x_test_hot_encode = hot_encode.transform(x_test_hot_encode)
    print("Converting hot-encoded x_test to dataframe...")
    x_test_hot_encode = pd.DataFrame(x_test_hot_encode, index=x_test.index, columns=hot_encode.get_feature_names_out())

    print("Joining x_train and hot-encoded x_train...")
    x_train = pd.concat([x_train, x_train_hot_encode], axis=1).drop(cat_cols, axis=1)

    print("Joining x_test and hot-encoded x_test...")
    x_test = pd.concat([x_test, x_test_hot_encode], axis=1).drop(cat_cols, axis=1)

    ## scaling and normalizing
    print("Scaleing x_train & x_test...")
    scale = StandardScaler()
    x_train = scale.fit_transform(x_train)
    x_test = scale.fit_transform(x_test)

    ## dimension reduction
    print("Reducing dimensions for x_train & x_test...")
    pca = PCA(0.95)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    print("Successfully completed all pre-processing required...")

    return{
        "x_train": x_train,
        "x_test": x_test,
        "y_train_binary": y_train_binary,
        "y_test_binary": y_test_binary
    }







