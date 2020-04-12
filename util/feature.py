import pandas as pd
import numpy as np
from datetime import datetime as dt
import uuid

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from toto_logger.logger import TotoLogger

logger = TotoLogger()

class FeatureEngineering: 

    def __init__(self): 
        pass

    def __cl_feature_engineering(self, data):
        """
        Feature engineering for the clustering model
        The clustering model finds the best time clusters given the distribution of meals over the time in the day
        """
        raw_data = pd.to_numeric(data['time'].str.split(':').str[0]) + pd.to_numeric(data['time'].str.split(':').str[1]) / 60
        
        return pd.DataFrame(raw_data, columns=['time']).round(2)

    def __cl_train(self, features, n_clusters): 
        
        km = KMeans(n_clusters=n_clusters)
        
        km.fit(features['time'].to_numpy().reshape(-1, 1))
        
        return km

    def __apply_clustering(self, data, clustering_model): 
        """
        Applies the clustering model to the time. 
        Assumes that the "time" column is a float: e.g. 07:30 will be 7.5
        That means that if the data is in the HH:mm format, then first apply the cl_feature_engineering() function
        """
        return data['time'].apply(lambda x : clustering_model.predict(np.array([[x]]))[0])

    def __encode_ids(self, data): 
        """
        Encodes the aliments ids
        """
        id_encoder = LabelEncoder()
        id_encoder.fit(data['id'])

        return (id_encoder, id_encoder.transform(data['id']))

    def __home(self, date):
        """
        Holds the logic of the staying home feature engineering func

        Parameters
        ----------
        date (datetime)
            The date

        Returns
        -------
        home (int)
            0 if it's False, 1 if True

        """
        if date >= dt(2020, 3, 12) and date <= dt(2020, 5, 10): 
            return 1

        if date.weekday() == 5:
            return 1
        if date.weekday() == 6:
            return 1
        
        return 0

    def __staying_home(self, data):
        """
        Defines if the user is staying home.
        There's a bit of hard coded logic here, waiting for an app that helps tracking if I'm home or not, etc..
        """
        return pd.to_datetime(data['date']).apply(self.__home)    
        

    def __back_to_one(self, row): 
        new_row = []
        for x in row: 
            if x > 1: 
                new_row.append(1)
            else:
                new_row.append(x)

        return new_row

    def do_for_predict(self, data, time_cluster_model, id_encoder, context): 
        """
        Engineers the features for the prediction

        Parameters
        ----------
        data (dict)
            Requires the following keys to be present: 
            - time (str, %H%M)
            - date (str, %Y%m%d)
        """
        time = data['time']
        date = data['date']

        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Starting feature engineering for request prediction ({time}, {date})'.format(context=context.process, time=time, date=date), 'info')

        # Convert the time into a float and find the time cluster
        time_float = int(time.split(':')[0]) + (int(time.split(':')[1])/60)
        time_cluster = time_cluster_model.predict([[time_float]])[0]

        # Define the staying home feature
        staying_home = self.__home(dt.strptime(date, '%Y%m%d'))

        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Engineered features for ({time}, {date}): [{tc}, {sh}]'.format(context=context.process, time=time, date=date, tc=time_cluster, sh=staying_home), 'info')

        return [[time_cluster, staying_home]]


    def do_for_train(self, folder, raw_data_filename, context): 
        """
        Engineers the features
        """
        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Starting feature engineering'.format(context=context.process), 'info')

        # Load the data
        raw_data_df = pd.read_csv(raw_data_filename)

        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Starting with a raw data shape of {s}'.format(context=context.process, s=raw_data_df.shape), 'info')

        # Creates the features to determine the clusters
        cl_features = self.__cl_feature_engineering(raw_data_df)

        # Define how many clusters we think there should be
        n_clusters = 7

        # Find the clusters
        time_cluster_model = self.__cl_train(cl_features, n_clusters)

        ###### TIME CLUSTERING APPLIED
        ###### NOW LET'S FILTER THE DATA, KEEPING ONLY THE MOST FREQUENTLY OCCURRING ALIMENTS --------------
        num_training_days = raw_data_df['date'].nunique()
        sorted_aliments = raw_data_df['id'].value_counts()

        sorted_aliments = pd.DataFrame(sorted_aliments).reset_index().rename(columns={"index": 'id', "id": 'count'})
        sorted_aliments['frequency'] = sorted_aliments['count'] / num_training_days

        filtered_aliments = sorted_aliments[sorted_aliments['frequency'] >= 0.15]['id']

        filtered_raw_data = raw_data_df[raw_data_df['id'].isin(filtered_aliments)]

        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Filtered raw data shape: {s}'.format(context=context.process, s=filtered_raw_data.shape), 'info')

        ####### DATA FILTERED
        ####### NOW LET'S DO THE FEATURE ENGINEERING ---------------------------------------------------------

        # Create the features to apply the time clustering to the data
        features_df = pd.concat([filtered_raw_data[['id', 'date']], self.__cl_feature_engineering(filtered_raw_data)], axis=1)
        
        # Apply the time clustering
        features_df['time_cluster'] = self.__apply_clustering(features_df, time_cluster_model)

        # Check if home
        features_df['staying_home'] = self.__staying_home(features_df)
        
        # Encode IDs
        (id_encoder, encoded_ids) = self.__encode_ids(features_df)
        
        features_df['encoded_id'] = encoded_ids
        
        # Drop some columns
        features_df.drop(columns=['date'], inplace=True)

        ######### BASIC FEATURE ENGINEERING DONE
        ######### NOW I HAVE TO FLATTEN THE STRUCTURE: EACH TIME NEEDS ONLY ONE ROW WITH ALL THE ALIMENTS EATEN AT THAT TIME
        # Create the dummy 
        features_wd = pd.concat([features_df, pd.get_dummies(features_df['encoded_id'])], axis=1)
        features_wd.drop(columns=['encoded_id'], inplace=True)

        # Group by time, staying home and cluster
        gb = features_wd.groupby(by=['time', 'staying_home', 'time_cluster'])

        # Sum all the remaining columns: the dummies of the ids! 
        features_flat = gb.sum()

        # Normalize: all rows counting an aliment > 1 => bring back to 1 so that it's only zeroes and ones
        features_flat_norm = features_flat.apply(self.__back_to_one).reset_index()

        # Save the features
        features_filename = '{folder}/features.{uid}.csv'.format(folder=folder, uid=uuid.uuid1());

        features_flat_norm.to_csv(features_filename)

        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Feature engineering completed. Features Shape: {r}'.format(context=context.process, r=features_flat_norm.shape), 'info')

        # Return the file and the vectorizer
        return (features_filename, time_cluster_model, id_encoder)

