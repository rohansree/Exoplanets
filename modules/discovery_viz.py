import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from geopy.geocoders import Nominatim

class Discovery_Viz():
    '''
    Class to visualize the discovery data

    Attributes:
        df (pd.DataFrame): the dataframe to be visualized
        theme (str): the theme of the visualization, default is 'plotly_dark'
        colors (list[str]): the list of colors for the visualization, default is ['#636EFA',  '#EF553B',  '#00CC96',  '#AB63FA',  '#FFA15A',  '#19D3F3',  '#FF6692',  '#B6E880',  '#FF97FF', '#FECB52']

    Examples:
            Sample usage of Discovery_Viz class:
            
                >>> from modules.discovery_viz import Discovery_Viz
                >>> discovery_viz = Discovery_Viz('data/NASA_planetary_data.csv')
                >>> discovery_viz.plot_telescope_pie()
                >>> discovery_viz.ground_discovery_map()
                >>> discovery_viz.discovery_facility_hist()
                >>> discovery_viz.year_locale_scatter()
                >>> discovery_viz.locale_hist()
                >>> discovery_viz.locale_hist('telescope')
                >>> discovery_viz.locale_hist('instrument')
    '''
    def __init__(self, df_location : str, theme : str = 'plotly_dark', colors : list[str] = ['#636EFA',  '#EF553B',  '#00CC96',  '#AB63FA',  '#FFA15A',  '#19D3F3',  '#FF6692',  '#B6E880',  '#FF97FF', '#FECB52']):
        '''
        The constructor for Discovery_Viz class.

        Parameters:
            df_location (str): the location of the dataframe to be visualized
            theme (str): the theme of the visualization, default is 'plotly_dark'
            colors (list[str]): the list of colors for the visualization, default is ['#636EFA',  '#EF553B',  '#00CC96',  '#AB63FA',  '#FFA15A',  '#19D3F3',  '#FF6692',  '#B6E880',  '#FF97FF', '#FECB52']

        Returns:
            None

        '''
        self.df = pd.read_csv(df_location, skiprows = 168)
        self.df['disc_locale'] = self.df['disc_locale'].replace('Multiple Locale', 'Multiple Locales')
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
        telescope_counts = pd.concat([top5_telescopes, telescope_others])

        # Select the top 5 instruments, and group the rest into 'Others'
        top5_instruments = instrument_counts.head(5)
        instrument_others = pd.Series(instrument_counts[5:].sum(), index=['Others'])
        instrument_counts = pd.concat([top5_instruments, instrument_others])

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
    

    def ground_discovery_map(self):
        '''
        Function to plot the map of the discovery facilities on the ground

        Returns:
            fig: the map of the discovery facilities on the ground
        '''

        # Get the coordinates of the discovery facilities on the ground
        ground_facilities = self.df[self.df['disc_locale'] == 'Ground'].copy().reset_index(drop=True)
        ground_fac_dict = {i:self.get_coordinates(i) for i in list(ground_facilities['disc_facility'].value_counts().index)}
        ground_fac_series = pd.Series(ground_fac_dict, name = 'local')

        # Create a dataframe of the discovery facilities on the ground
        df_disc = pd.concat([ground_fac_series, ground_facilities['disc_facility'].value_counts()], axis=1).reset_index().dropna()
        # Rename the columns
        df_disc['latitude'] = df_disc.apply(lambda row: row.local[0], axis = 1)
        df_disc['longitude'] = df_disc.apply(lambda row: row.local[1], axis = 1)

        # Create a scatter plot of the discovery facilities on the ground
        fig = px.scatter_geo(
                df_disc,
                lat='latitude',
                lon='longitude',
                title='Discovery Facilities on the Ground',
                hover_name='index',
                projection='natural earth',
                size = 'count'
            )

        # Show the plot
        return fig
    
    def discovery_facility_hist(self, count_limit : int = 50):
        '''
        Function to plot the histogram of the discovery facilities

        Parameters:
            count_limit (int): the count limit of the discovery facilities, default is 50

        Returns:
            fig: the histogram of the discovery facilities
        '''
        assert count_limit > 0, 'count_limit must be greater than 0'
        # Get the count of the discovery facilities
        df_mod = self.df['disc_facility'].value_counts()
        df_mod = self.df[self.df.disc_facility.isin(list(df_mod[df_mod > 50].index))]

        # Create a histogram of the discovery facilities
        fig2 = px.bar(df_mod['disc_facility'].value_counts().reset_index(), x='disc_facility', y='count',
              labels={'disc_facility': 'Discovery Facility'},
              title=f'Count of Discovery Facility with > {count_limit} counts', template = self.theme)
        
        return fig2

    def year_locale_scatter(self):
        '''
        Function to plot the scatter plot of the year and locale of the discovery

        Returns:
            fig: the scatter plot of the year and locale of the discovery
        '''

        # Create a scatter plot of the year and locale of the discovery
        fig = px.scatter(self.df, x='disc_year', y='disc_locale', color='disc_locale',
                  labels={'disc_year': 'Discovery Year', 'disc_locale': 'Discovery Locale'},
                  title='Relationship between Discovery Year and Locale', template = self.theme)
        
        fig.update_traces(marker=dict(size=10))  # Adjust the size as needed
        # Increase the size of y-axis labels
        fig.update_layout(yaxis=dict(tickfont=dict(size=18)))  # Adjust the size as needed

        return fig
    
    def locale_hist(self, color_select : str = 'locale'):
        '''
        Function to plot the histogram of the discovery method and locale

        Parameters:
            color_select (str): the color selection and it can have the values ['locale', 'telescope', 'instrument'], default is 'locale'

        Returns:
            fig: the histogram of the discovery method and locale
        '''

        assert color_select in ['locale', 'telescope', 'instrument'], 'color_select must be either "locale" or "method"'
        match color_select:

            case 'locale':
                # Create a histogram of the discovery method and locale
                fig = px.histogram(self.df, x='disc_locale', color='discoverymethod',
                        labels={'disc_locale': 'Discovery Locale', 'discoverymethod': 'Discovery Method'},
                        title='Discovery Method and Locale', template = self.theme)
                
            case 'telescope':
                # Create a histogram of the discovery method and locale
                fig = px.histogram(self.df, x='disc_locale', color='disc_telescope',
                        labels={'disc_locale': 'Discovery Locale', 'disc_telescope': 'Discovery Telescope'},
                        title='Discovery Telescope and Locale', template = self.theme)
            
            case 'instrument':
                # Create a histogram of the discovery method and locale
                fig = px.histogram(self.df, x='disc_locale', color='disc_instrument',
                        labels={'disc_locale': 'Discovery Locale', 'disc_instrument': 'Discovery Instrument'},
                        title='Discovery Instrument and Locale', template = self.theme)
        
        return fig
    
    @staticmethod
    def get_coordinates(location_name:str)->tuple[float, float]:
        '''
        Function to get the coordinates of a location

        Parameters:
            location_name (str): the name of the location

        Returns:
            tuple: the latitude and longitude of the location
        '''
        geolocator = Nominatim(user_agent="my_geocoder")
        location = geolocator.geocode(location_name)

        if location:
            return location.latitude, location.longitude
        else:
            return None, None
        

if __name__ == "__main__":
    discovery_viz = Discovery_Viz('data/NASA_planetary_data.csv')
    discovery_viz.plot_telescope_pie()
    discovery_viz.ground_discovery_map()
    discovery_viz.discovery_facility_hist()
    discovery_viz.year_locale_scatter()
    discovery_viz.locale_hist()
    discovery_viz.locale_hist('telescope')
    discovery_viz.locale_hist('instrument')
