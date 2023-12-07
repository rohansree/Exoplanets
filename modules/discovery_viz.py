import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from geopy.geocoders import Nominatim


class Discovery_Viz():
    '''
    Class to visualize the discovery data
    '''
    def __init__(self, df_location : str, theme : str = 'plotly_dark', colors : list[str] = ['#636EFA',  '#EF553B',  '#00CC96',  '#AB63FA',  '#FFA15A',  '#19D3F3',  '#FF6692',  '#B6E880',  '#FF97FF', '#FECB52']):
        '''
        The constructor for Discovery_Viz class.

        Parameters:
            df_location (str): the location of the dataframe to be visualized
            theme (str): the theme of the visualization, default is 'plotly_dark'
            colors (List[str]): the list of colors for the visualization, default is ['#636EFA',  '#EF553B',  '#00CC96',  '#AB63FA',  '#FFA15A',  '#19D3F3',  '#FF6692',  '#B6E880',  '#FF97FF', '#FECB52']
        '''
        self.df = pd.read_csv(df_location, skiprows = 168)
        self.theme = theme
        if theme == 'plotly_dark':
            plt.style.use('dark_background')

        self.colors = colors

    def plot_telescope_pie(self):

        '''
        Function to plot the pie chart of the distribution of telescopes

        Returns:
            plt: the pie chart of the distribution of telescopes
        '''

        # Count the occurrences of each telescope and instrument
        telescope_counts = self.df['disc_telescope'].value_counts()
        instrument_counts = self.df['disc_instrument'].value_counts()

        # Select the top 5 telescopes, and group the rest into 'Others'
        top5_telescopes = telescope_counts.head(5)
        telescope_others = pd.Series(telescope_counts[5:].sum(), index=['Others'])
        telescope_counts = top5_telescopes.append(telescope_others)

        # Select the top 5 instruments, and group the rest into 'Others'
        top5_instruments = instrument_counts.head(5)
        instrument_others = pd.Series(instrument_counts[5:].sum(), index=['Others'])
        instrument_counts = top5_instruments.append(instrument_others)

        # Create a pie chart for telescopes
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)  # Subplot for telescopes
        plt.pie(telescope_counts, labels=telescope_counts.index, autopct='%1.1f%%', startangle=90, colors= self.colors)
        plt.title('Distribution of Telescopes')

        # Create a pie chart for instruments
        plt.subplot(1, 2, 2)  # Subplot for instruments
        plt.pie(instrument_counts, labels=instrument_counts.index, autopct='%1.1f%%', startangle=90, colors = self.colors)
        plt.title('Distribution of Instruments')

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        return plt
    
    @staticmethod
    def get_coordinates(location_name):
        geolocator = Nominatim(user_agent="my_geocoder")
        location = geolocator.geocode(location_name)

        if location:
            return location.latitude, location.longitude
        else:
            return None, None
