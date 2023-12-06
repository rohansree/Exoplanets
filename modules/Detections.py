import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Detections:
    '''
    The Detections class is a class that visualizes the exoplanet detection data.

    Attributes:
        df (DataFrame): the exoplanet dataframe to be visualized
        

    '''
    def __init__(self, df_location):
        '''
        The constructor for Detections class.

        Parameters:
            df_location (str): the location of the dataframe to be visualized
        '''
        self.df = pd.read_csv(df_location, skiprows = 168)
        # df = pd.read_csv('NASA_planetary_data.csv', skiprows = 168)


    def detection_vis(self):
        """
        Args:
            df: the exoplanet dataframe to be visualized
        Returns:
            None

        Given the dataframe of exoplanets, 
        break it down and visualize each detection category.
        Goes through different categories and for eachs, 
        calls a function to plot it versus year of discovery
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
        for idx, flag in enumerate(cat_flags):

            name = cat_names[idx]
            self.detection_plot_single(flag, name)

            #so there is enough time for each category to be displayed on the screen
            time.sleep(300)
            print(idx)
            


    def detection_plot_single(self, flag, name):
        """
        Args:
            df: dataframe containing the detections dataset
            flag: the specific category of detections to visualize
            name: the longform name of flag, to be used for plot title and legend

        Returns:
            None

        Used to visualize individual, single columns of the detections dataset.
        """
        
        group = self.df.groupby(['disc_year', flag]).size().unstack()

        #colors to be used if plotting multiple flags together
        #this function only plots a single flag but kept for consistency
        #Single variable plot wil; just use only the first color
        colors = ['#636EFA',  '#EF553B',  '#00CC96',  '#AB63FA',  '#FFA15A',  '#19D3F3',  '#FF6692',  '#B6E880',  '#FF97FF',  '#FECB52']

        #plot bar chart
        ax = group.plot(kind='bar', stacked=True, figsize=(12,6), color=colors)#['#90EE90'])#, hue_order=[1,0])

        plt.ylabel('Planets Detected')
        plt.xlabel('Discovery Year')

        title = "Planets Detected via "
        title += name
        title += ", 1992-2023"

        leg = "Number of Planets Detected via "
        leg += name
        
        plt.title(title)
        plt.legend([leg])
        plt.show()