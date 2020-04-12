from pandas import json_normalize

from toto_logger.logger import TotoLogger

from remote.diet import DietAPI

logger = TotoLogger()

class HistoryDownloader: 

    def __init__(self, train_from): 
        """
        Parameters
        ----------
        train_from (str)
            The date from where to train, in a %Y%m%d format
        """
        self.train_from = train_from

    def download(self, folder, context):
        """
        Downloads the history of meals to the specified folder
        """
        dateGte = self.train_from

        logger.compute(context.correlation_id, '[ {context} ] - [ HISTORICAL ] - Starting historical data download from date {date}'.format(context=context.process, date=dateGte), 'info')

        history_filename = '{folder}/history-from-{dateGte}.csv'.format(folder=folder, dateGte=dateGte);

        # Download
        json_data = DietAPI().get_meals(dateGte, context.correlation_id)

        # Extract the mealls array
        try: 
            raw_data_df = json_normalize(json_data['meals'], 'aliments', ['date', 'time'])
        except: 
            logger.compute(context.correlation_id, '[ {context} ] - [ HISTORICAL ] - Error reading the following microservice response: {r}'.format(context=context.process, r=json_data), 'error')
            logger.compute(context.correlation_id, '[ {context} ] - [ HISTORICAL ] - No historical data'.format(context=context.process), 'warn')
            return None

        # Drop useless data
        raw_data_df.drop(['amountGr', 'amountMl', 'amount'], axis=1, inplace=True)
        
        # Save 
        raw_data_df.to_csv(history_filename) 

        logger.compute(context.correlation_id, '[ {context} ] - [ HISTORICAL ] - Historical data downloaded: {r} rows'.format(context=context.process, r=len(raw_data_df)), 'info')

        return history_filename