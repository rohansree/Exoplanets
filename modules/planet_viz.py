import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class Planet_Viz:
    '''
    The Planet_Viz class is a class that visualizes the exoplanet dataframe.

    Attributes:
        df (DataFrame): the exoplanet dataframe to be visualized
        df_loc (DataFrame): the exoplanet dataframe to be visualized with only the location data
        theme (str): the theme of the visualization, default is 'plotly_dark'
        colors (List[str]): the list of colors for the visualization, default is ['#636EFA',  '#EF553B',  '#00CC96',  '#AB63FA',  '#FFA15A',  '#19D3F3',  '#FF6692',  '#B6E880',  '#FF97FF', '#FECB52']


    Example:
        A simple example of how to use this class is:

            >>> from modules.planet_viz import Planet_Viz
            >>> pv = Planet_Viz('data/NASA_planetary_data.csv')
            >>> pv.interative_planet_viz()
            >>> pv.equivolume_bins_histogram()
            >>> pv.planet_yr_method_dist()
            >>> pv.planet_yr_method_hist()
            >>> pv.detection_vis_combined()
            >>> pv.detection_vis_combined(remove_transit = True)
            >>> pv.detection_vis_separate()

    '''
    def __init__(self, df_location : str, theme : str = 'plotly_dark', colors : list[str] = ['#636EFA',  '#EF553B',  '#00CC96',  '#AB63FA',  '#FFA15A',  '#19D3F3',  '#FF6692',  '#B6E880',  '#FF97FF', '#FECB52']):
        '''
        The constructor for Planet_Viz class.

        Parameters:
            df_location (str): the location of the dataframe to be visualized
            theme (str): the theme of the visualization, default is 'plotly_dark'
            colors (List[str]): the list of colors for the visualization, default is ['#636EFA',  '#EF553B',  '#00CC96',  '#AB63FA',  '#FFA15A',  '#19D3F3',  '#FF6692',  '#B6E880',  '#FF97FF', '#FECB52']
        '''
        df = pd.read_csv(df_location, skiprows = 168)
        self.theme = theme
        if theme == 'plotly_dark':
            plt.style.use('dark_background')

        self.colors = colors
        self.df = df
        self.df_loc = df[['pl_name','disc_year', 'ra', 'dec', 'sy_dist', 'discoverymethod']].copy()
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
            fig (go.Figure): The interactive figure of the dataframe
        
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
        return fig

    @staticmethod
    def calc_radii_lst(max_dist : int = 1500, vol : int = 20000000, start : int = 0):
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
            fig (go.Figure): The equivolume bins histogram figure of the dataframe
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

        return fig

    
    def planet_yr_method_dist(self):
        '''
        The planet year method distance visualization of Planet_Viz class.

        Returns:
            fig (go.Figure): The planet year method distance figure of the dataframe

        '''
        fig = px.scatter(self.df_loc, y = 'sy_dist', x='disc_year', color='discoverymethod', labels={
                     "sy_dist": "Distance from Earth (parsec)",
                     "disc_year": "Discovered Year",
                     "discoverymethod": "Method of Discovery"
                 }, template = self.theme, title='Distance from Earth vs. Discovered Year by Method of Discovery')
        return fig

    def planet_yr_method_hist(self, cumulative : bool = True):

        '''
        The planet year method histogram visualization of Planet_Viz class.

        Parameters:
            cumulative (bool): whether the histogram is cumulative or not, default is True
        
        Returns:
            fig (go.Figure): The planet year method histogram figure of the dataframe
        '''
        fig = px.histogram(self.df_loc, x='disc_year', cumulative=cumulative, color = 'discoverymethod', labels={
                     "disc_year": "Discovered Year",
                     "discoverymethod": "Method of Discovery"
                 }, template = self.theme, title=f'Histogram of Planet Counts by Year with Discovery Method (Cumulative = {cumulative})').update_layout(yaxis_title = "Number of Planets")
        return fig



    def detection_vis_combined(self, remove_transit : bool = False):

        '''
        The visualization of the exoplanet detection data for all methods combined.
        Args:
            df (Datafrane): the exoplanet dataframe to be visualized
            remove_transit (bool): boolean to remove transit method from visualization
        Returns:
            plot (Plot): plot of the detections dataset for all methods combined  
        '''

        if remove_transit:
            group = self.df[self.df['discoverymethod'] != 'Transit'].groupby(['disc_year', 'discoverymethod']).size().unstack()
        else:
            group = self.df.groupby(['disc_year', 'discoverymethod']).size().unstack()

        #plot bar chart
        # ax = group11.plot(kind='bar', stacked=True, figsize=(12,6))#, color=['#bc5090','#90EE90'])#, hue_order=[1,0])
        #plot bar chart
        ax = group.plot(kind='bar', stacked=True, figsize=(12,6), color=self.colors)#['#90EE90'])#, hue_order=[1,0])


        plt.xlabel('Discovery Year')
        plt.ylabel('New Planets Discovered')
        if remove_transit:
            plt.title('Planet Discovery Method (Transit Removed)')
        else:
            plt.title('Planet Discovery Method')

        return plt
                     
                    
    def detection_vis_separate(self):
        """   
        Given the dataframe of exoplanets, 
        break it down and visualize each detection category.
        Goes through different categories and for eachs, 
        calls a function to plot it versus year of discovery

        Args:
            df (Dataframe): the exoplanet dataframe to be visualized
        Returns:
            plot_list (List[Plot]): list of plots, one for each category
        """

        #column names in dataframe
        cat_flags = ['rv_flag',
                    'pul_flag',
                    'ptv_flag',
                    'tran_flag',
                    'ast_flag',
                    'obm_flag',
                    'micro_flag',
                    'etv_flag',
                    'ima_flag',
                    'dkin_flag'
        ]
        
        #longform category names, for title
        cat_names = ['Radial Velocity Variations',
                    'Pulsar Timing Variations',
                    'Pulsation Timing Variations',
                    'Transits',
                    'Astrometric Variations',
                    'Orbital Brightness Variations',
                    'Microlensing',
                    'Eclipse Timing Variations',
                    'Imaging',
                    'Disk Kinematics'
        ]

        
        #go through every flag in the detections category
        plot_list = []
        for idx, flag in enumerate(cat_flags):

            name = cat_names[idx]
            plot_list.append(self.detection_plot_single(flag, name))
        return plot_list
            


    def detection_plot_single(self, flag : str , name : str ):
        """
        Used to visualize individual, single columns of the detections dataset.

        Args:
            df (Dataframe): dataframe containing the detections dataset
            flag (str): the specific category of detections to visualize
            name (str): the longform name of flag, to be used for plot title and legend

        Returns:
            plot (Plot): plot of the detections dataset for the given flag
        """
        
        #group by year and flag, count number of planets detected

        group = self.df.groupby(['disc_year', flag]).size().unstack()
        group = group.fillna(0)
        group = group.astype(int)
        group = group.rename(columns={0: "Not Detected", 1: "Detected"})
        group = group.drop(columns=['Not Detected'])


        #colors to be used if plotting multiple flags together
        #this function only plots a single flag but kept for consistency
        #Single variable plot wil; just use only the first color
    
        #plot bar chart
        ax = group.plot(kind='bar', figsize=(12,6), color=self.colors)#['#90EE90'])#, hue_order=[1,0])

        plt.ylabel('Planets Detected')
        plt.xlabel('Discovery Year')

        title = "Planets Detected via "
        title += name
        title += ", 1992-2023"

        leg = "Number of Planets Detected via "
        leg += name
        
        plt.title(title)
        plt.legend([leg])
        return plt
    
if __name__ == "__main__":
    pv = Planet_Viz('data/NASA_planetary_data.csv')
    pv.interative_planet_viz()
    pv.equivolume_bins_histogram()
    pv.planet_yr_method_dist()
    pv.planet_yr_method_hist()