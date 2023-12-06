import pandas as pd
import numpy as np
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objects as go

class Planet_Viz:
    '''
    The Planet_Viz class is a class that visualizes the exoplanet dataframe.

    Attributes:
        df_loc (DataFrame): the exoplanet dataframe to be visualized
        theme (str): the theme of the visualization, default is 'plotly_dark'

    Example:
        A simple example of how to use this class is:

            >>> from modules.planet_viz import Planet_Viz
            >>> pv = Planet_Viz('data/NASA_planetary_data.csv')
            >>> pv.interative_planet_viz()
            >>> pv.equivolume_bins_histogram()
            >>> pv.planet_yr_method_dist()
            >>> pv.planet_yr_method_hist()

    '''
    def __init__(self, df_location : str, theme : str = 'plotly_dark'):
        '''
        The constructor for Planet_Viz class.

        Parameters:
            df_location (str): the location of the dataframe to be visualized
            theme (str): the theme of the visualization, default is 'plotly_dark'
        '''
        df = pd.read_csv(df_location, skiprows = 168)
        self.theme = theme
        self.df_loc = df[['pl_name','disc_year', 'ra', 'dec', 'sy_dist', 'discoverymethod']]
        self.df_loc = self.df_loc.sort_values(by=['discoverymethod'])
        self.df_loc['x'], self.df_loc['y'], self.df_loc['z'] = self.df_loc['sy_dist'] * np.cos(self.df_loc['ra']) * np.cos(self.df_loc['dec']), self.df_loc['sy_dist'] * np.sin(self.df_loc['ra']) * np.cos(self.df_loc['dec']), self.df_loc['sy_dist'] * np.sin(self.df_loc['dec'])

    def __repr__(self):
        '''
        The representation of Planet_Viz class.

        Returns:
            dataframe.head: The first 5 rows of the dataframe
        '''
        return self.df_loc.head()

    def interative_planet_viz(self, dist_limit : int = 1000) -> go.Figure:
        '''
        The interactive visualization of Planet_Viz class.

        Parameters:
            dist_limit (int): the distance limit of the visualization, default is 1000

        Returns:
            fig: The interactive figure of the dataframe
        
        '''

        assert dist_limit > 0, "dist_limit must be greater than 0"

        #create a new modified dataframe with distance limit and the earth at the origin
        df_loc_mod = self.df_loc[self.df_loc['sy_dist'] < dist_limit].copy()
        df_loc_mod = pd.concat([pd.DataFrame([['earth', 0, 0, 0, 0, 0, 0, 0, 0]], columns=df_loc_mod.columns), df_loc_mod], ignore_index=True)

        column_list = list(df_loc_mod.columns) + ['cum_year']


        #make a new dataframe with the cumulative year
        data = []

        for i in range(1992, 2024):
            tmp = df_loc_mod[df_loc_mod['disc_year']<= i].copy()
            tmp['cum_year'] = i
            data += tmp.values.tolist()

        df_new = pd.DataFrame(data, columns = column_list)

        #create a color map for the discovery year
        n_colors = 2024-1992
        colors = px.colors.sample_colorscale("sunset", [n/(n_colors -1) for n in range(n_colors)])
        color_map = { str(i+1992) : x for i,x in enumerate(colors)} | { str(0) : 'blue'}

        #convert the discovery year to string for the color map
        df_new['disc_year'] = df_new["disc_year"].astype(str)

        #concat a new dataframe with empty values and away from the x,y,z limits to add years to the legend
        df_new = pd.concat([pd.DataFrame([['x'+str(i), str(i), 0, 0, 0, dist_limit*1.1, 0, 0, 1992, str(1992)] for i in range(1992, 2024)], columns=df_new.columns), df_new], ignore_index=True)

        #plotly figure
        fig = px.scatter_3d(df_new, x='x', y='y', z='z', color='disc_year', hover_name='pl_name', hover_data = {k : k in ['x', 'y', 'z'] for k in df_new.columns}, animation_frame= 'cum_year',color_discrete_map=color_map, range_x=[-dist_limit, dist_limit], range_y = [-dist_limit, dist_limit], range_z = [-dist_limit, dist_limit], labels={'disc_year':'Discovery Year', 'cum_year':'Planets Discovered Till'}, template = self.theme)
        fig.update_scenes(aspectmode='cube')
        fig.show()

    def __calc_radii_lst(self, max_dist : int = 1500, vol : int = 20000000, start : int = 0):
        ''' 
        Helper function for equivolume_bins_histogram

        Parameters:
            max_dist (int): the maximum distance from earth
            vol (int): the volume of each bin
            start (int): the starting radius from earth of the histogram
        
        Returns:
            radii_lst (list): A list of radii with adjacent pairs having a volume of vol
        '''

        radii_lst = [start]
        r_prev = radii_lst[-1]
        while r_prev < max_dist-1:
            radii_lst.append((vol+r_prev**3)**(1./3.))
            r_prev = radii_lst[-1]
        return radii_lst

    def equivolume_bins_histogram(self, rad_start : int = 400 , vol : int = 20000000, max_dist : int = 1500) -> go.Figure:
        '''
        The equivolume bins histogram visualization of Planet_Viz class.

        Parameters:
            rad_start (int): the starting radius from earth of the histogram, default is 400
            vol (int): the volume of each bin, default is 20000000
            max_dist (int): the maximum distance from earth, default is 1500

        Returns:
            fig: The equivolume bins histogram figure of the dataframe
        '''
        assert rad_start > 0, "rad_start must be greater than 0"
        assert vol > 0, "vol must be greater than 0"

        #create a list of radii with adjacent pairs having a volume of vol
        bins1 = self.__calc_radii_lst(start = rad_start, vol = vol, max_dist = max_dist)

        #create a list of bins with the same width as the radii list
        counts, bins2 = np.histogram(self.df_loc.sy_dist, bins=bins1)
        bins2 = 0.5 * (bins1[:-1] + bins2[1:])

        #create a list of widths for the histogram
        widths = []
        for i, b1 in enumerate(bins1[1:]):
            widths.append((b1-bins2[i])*2)

        #plotly figure
        fig = go.Figure(go.Bar(x=bins2,y=counts,width=widths))\
            .update_layout(title = 'Histogram of Exoplanets Discovered for Equivolume Bins', xaxis_title = "Distance from Earth (parsec)",  yaxis_title = "Number of Planets", template = self.theme)

        fig.show()

    
    def planet_yr_method_dist(self):
        '''
        The planet year method distance visualization of Planet_Viz class.

        Returns:
            fig: The planet year method distance figure of the dataframe

        '''
        fig = px.scatter(self.df_loc, y = 'sy_dist', x='disc_year', color='discoverymethod', labels={
                     "sy_dist": "Distance from Earth (parsec)",
                     "disc_year": "Discovered Year",
                     "discoverymethod": "Method of Discovery"
                 }, template = self.theme, title='Distance from Earth vs. Discovered Year by Method of Discovery')
        fig.show()

    def planet_yr_method_hist(self, cumulative : bool = True):

        '''
        The planet year method histogram visualization of Planet_Viz class.

        Parameters:
            cumulative (bool): whether the histogram is cumulative or not, default is True
        
        Returns:
            fig: The planet year method histogram figure of the dataframe
        '''
        fig = px.histogram(self.df_loc, x='disc_year', cumulative=cumulative, color = 'discoverymethod', labels={
                     "disc_year": "Discovered Year",
                     "discoverymethod": "Method of Discovery"
                 }, template = 'plotly_dark', title=f'Histogram of Planet Counts by Year with Discovery Method (Cumulative = {cumulative})').update_layout(yaxis_title = "Number of Planets")
        fig.show()


if __name__ == "__main__":
    pv = Planet_Viz('data/NASA_planetary_data.csv')
    pv.interative_planet_viz()
    pv.equivolume_bins_histogram()
    pv.planet_yr_method_dist()
    pv.planet_yr_method_hist()