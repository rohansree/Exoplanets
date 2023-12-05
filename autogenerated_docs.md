# modules package

## Submodules

## modules.detections module

### modules.detections.detection_plot_single(df, flag, name)

@param df: dataframe containing the detections dataset
@param flag: the specific category of detections to visualize
@param name: the longform name of flag, to be used for plot title and legend

@return none

Used to visualize individual, single columns of the detections dataset.

### modules.detections.detection_vis(df)

@param df: the exoplanet dataframe to be visualized
@return none

Given the dataframe of exoplanets,
break it down and visualize each detection category.
Goes through different categories and for eachs,
calls a function to plot it versus year of discovery

### modules.detections.read_data()

Read and format dataframe from .csv file and call the visualization method,
passing the formatted dataframe to it

This function is separated because detections are only part of the dataset,
and the csv will only need to be read once.
* **Parameters:**
  **None** – 
* **Returns:**
  None

### Examples

A simple example of how to use this function is:

```pycon
read_data()
```

## modules.planet_viz module

### *class* modules.planet_viz.Planet_Viz(df_location: str, theme: str = 'plotly_dark')

Bases: `object`

The Planet_Viz class is a class that visualizes the exoplanet dataframe.

#### df_loc

the exoplanet dataframe to be visualized
* **Type:**
  DataFrame

#### theme

the theme of the visualization, default is ‘plotly_dark’
* **Type:**
  str

### Example

A simple example of how to use this class is:

```pycon
from modules.planet_viz import Planet_Viz
pv = Planet_Viz('data/NASA_planetary_data.csv')
pv.interative_planet_viz()
pv.equivolume_bins_histogram()
pv.planet_yr_method_dist()
pv.planet_yr_method_hist()
```

#### \_\_init_\_(df_location: str, theme: str = 'plotly_dark')

The constructor for Planet_Viz class.
* **Parameters:**
  * **df_location** (*str*) – the location of the dataframe to be visualized
  * **theme** (*str*) – the theme of the visualization, default is ‘plotly_dark’

#### equivolume_bins_histogram(rad_start: int = 400, vol: int = 20000000)

The equivolume bins histogram visualization of Planet_Viz class.
* **Parameters:**
  * **rad_start** (*int*) – the starting radius from earth of the histogram, default is 400
  * **vol** (*int*) – the volume of each bin, default is 20000000
* **Returns:**
  The equivolume bins histogram figure of the dataframe
* **Return type:**
  fig

#### interative_planet_viz()

The interactive visualization of Planet_Viz class.
* **Returns:**
  The interactive figure of the dataframe
* **Return type:**
  fig

#### planet_yr_method_dist()

The planet year method distance visualization of Planet_Viz class.
* **Returns:**
  The planet year method distance figure of the dataframe
* **Return type:**
  fig

#### planet_yr_method_hist(cumulative: bool = True)

The planet year method histogram visualization of Planet_Viz class.
* **Parameters:**
  **cumulative** (*bool*) – whether the histogram is cumulative or not, default is True
* **Returns:**
  The planet year method histogram figure of the dataframe
* **Return type:**
  fig
