import os
import pandas as pd
import requests

from toto_logger.logger import TotoLogger

logger = TotoLogger()

toto_auth = os.environ['TOTO_API_AUTH']
toto_host = os.environ['TOTO_HOST']

class DietAPI: 

    def __init__(self):
        pass

    def get_meals(self, dateGte, correlation_id): 
        """
        Retrieves the meals from Toto Diet API

        Parameters
        ----------
        dateGte (str)
            The date from which the meals should be downloaded.
            The date is a string in a %Y%m%d format

        """
        response = requests.get(
            'https://{host}/apis/diet/meals?&dateFrom={dateGte}'.format(dateGte=dateGte, host=toto_host),
            headers={
                'Accept': 'application/json',
                'Authorization': toto_auth,
                'x-correlation-id': correlation_id
            }
        )

        # Convert to JSON
        return response.json()
