import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data():
    """

    Read and format dataframe from .csv file and call the visualization method,
    passing the formatted dataframe to it

    This function is separated because detections are only part of the dataset, 
    and the csv will only need to be read once.

    Args:
        None
    Returns:
        None


    Examples:
        A simple example of how to use this function is:

            >>> read_data()
    
    """

    #only necessary on google colab
        # from google.colab import drive
        # drive.mount('/content/gdrive')
        # df = pd.read_csv('/content/gdrive/MyDrive/ECE 143 Group 18 Project/NASA_planetary_data.csv', skiprows = 168)

    #df stands for DataFrame

    df = pd.read_csv('data/NASA_planetary_data.csv', skiprows = 168)
    
    #to see what the data looks like
    # print(df.head())

    #calls the visualization method
    detection_vis(df)



def detection_vis(df):
    """
    @param df: the exoplanet dataframe to be visualized
    @return none

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
        detection_plot_single(df, flag, name)

        #so there is enough time for each category to be displayed on the screen
        time.sleep(300)
        print(idx)
        


def detection_plot_single(df, flag, name):
    """
    @param df: dataframe containing the detections dataset
    @param flag: the specific category of detections to visualize
    @param name: the longform name of flag, to be used for plot title and legend

    @return none

    Used to visualize individual, single columns of the detections dataset.
    """
    
    group = df.groupby(['disc_year', flag]).size().unstack()

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


#read_data()