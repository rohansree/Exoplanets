# modules package

## Submodules

## modules.discovery_viz module

### *class* modules.discovery_viz.Discovery_Viz(df_location: str, theme: str = 'plotly_dark', colors: list[str] = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'])

Bases: `object`

Class to visualize the discovery data

#### \_\_init_\_(df_location: str, theme: str = 'plotly_dark', colors: list[str] = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'])

The constructor for Discovery_Viz class.
* **Parameters:**
  * **df_location** (*str*) – the location of the dataframe to be visualized
  * **theme** (*str*) – the theme of the visualization, default is ‘plotly_dark’
  * **colors** (*List**[**str**]*) – the list of colors for the visualization, default is [‘#636EFA’,  ‘#EF553B’,  ‘#00CC96’,  ‘#AB63FA’,  ‘#FFA15A’,  ‘#19D3F3’,  ‘#FF6692’,  ‘#B6E880’,  ‘#FF97FF’, ‘#FECB52’]

#### *static* get_coordinates(location_name)

#### plot_telescope_pie()

Function to plot the pie chart of the distribution of telescopes
* **Returns:**
  the pie chart of the distribution of telescopes
* **Return type:**
  plt

## modules.model module

## modules.planet_viz module

### *class* modules.planet_viz.Planet_Viz(df_location: str, theme: str = 'plotly_dark', colors: list[str] = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'])

Bases: `object`

The Planet_Viz class is a class that visualizes the exoplanet dataframe.

#### df

the exoplanet dataframe to be visualized
* **Type:**
  DataFrame

#### df_loc

the exoplanet dataframe to be visualized with only the location data
* **Type:**
  DataFrame

#### theme

the theme of the visualization, default is ‘plotly_dark’
* **Type:**
  str

#### colors

the list of colors for the visualization, default is [‘#636EFA’,  ‘#EF553B’,  ‘#00CC96’,  ‘#AB63FA’,  ‘#FFA15A’,  ‘#19D3F3’,  ‘#FF6692’,  ‘#B6E880’,  ‘#FF97FF’, ‘#FECB52’]
* **Type:**
  List[str]

### Example

A simple example of how to use this class is:

```pycon
from modules.planet_viz import Planet_Viz
pv = Planet_Viz('data/NASA_planetary_data.csv')
pv.interative_planet_viz()
pv.equivolume_bins_histogram()
pv.planet_yr_method_dist()
pv.planet_yr_method_hist()
pv.detection_vis_combined()
pv.detection_vis_combined(remove_transit = True)
pv.detection_vis_separate()
```

#### \_\_init_\_(df_location: str, theme: str = 'plotly_dark', colors: list[str] = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'])

The constructor for Planet_Viz class.
* **Parameters:**
  * **df_location** (*str*) – the location of the dataframe to be visualized
  * **theme** (*str*) – the theme of the visualization, default is ‘plotly_dark’
  * **colors** (*List**[**str**]*) – the list of colors for the visualization, default is [‘#636EFA’,  ‘#EF553B’,  ‘#00CC96’,  ‘#AB63FA’,  ‘#FFA15A’,  ‘#19D3F3’,  ‘#FF6692’,  ‘#B6E880’,  ‘#FF97FF’, ‘#FECB52’]

#### *static* calc_radii_lst(max_dist: int = 1500, vol: int = 20000000, start: int = 0)

Helper function for equivolume_bins_histogram
* **Parameters:**
  * **max_dist** (*int*) – the maximum distance from earth
  * **vol** (*int*) – the volume of each bin
  * **start** (*int*) – the starting radius from earth of the histogram
* **Returns:**
  A list of radii with adjacent pairs having a volume of vol
* **Return type:**
  radii_lst (list)

#### detection_plot_single(flag: str, name: str)

Used to visualize individual, single columns of the detections dataset.
* **Parameters:**
  * **df** (*Dataframe*) – dataframe containing the detections dataset
  * **flag** (*str*) – the specific category of detections to visualize
  * **name** (*str*) – the longform name of flag, to be used for plot title and legend
* **Returns:**
  plot of the detections dataset for the given flag
* **Return type:**
  plot (Plot)

#### detection_vis_combined(remove_transit: bool = False)

The visualization of the exoplanet detection data for all methods combined.
:param df: the exoplanet dataframe to be visualized
:type df: Datafrane
:param remove_transit: boolean to remove transit method from visualization
:type remove_transit: bool
* **Returns:**
  plot of the detections dataset for all methods combined
* **Return type:**
  plot (Plot)

#### detection_vis_separate()

Given the dataframe of exoplanets,
break it down and visualize each detection category.
Goes through different categories and for eachs,
calls a function to plot it versus year of discovery
* **Parameters:**
  **df** (*Dataframe*) – the exoplanet dataframe to be visualized
* **Returns:**
  list of plots, one for each category
* **Return type:**
  plot_list (List[Plot])

#### equivolume_bins_histogram(rad_start: int = 400, vol: int = 20000000, max_dist: int = 1500)

The equivolume bins histogram visualization of Planet_Viz class.
* **Parameters:**
  * **rad_start** (*int*) – the starting radius from earth of the histogram, default is 400
  * **vol** (*int*) – the volume of each bin, default is 20000000
  * **max_dist** (*int*) – the maximum distance from earth, default is 1500
* **Returns:**
  The equivolume bins histogram figure of the dataframe
* **Return type:**
  fig (go.Figure)

#### interative_planet_viz(dist_limit: int = 1000)

The interactive visualization of Planet_Viz class.
* **Parameters:**
  **dist_limit** (*int*) – the distance limit of the visualization, default is 1000
* **Returns:**
  The interactive figure of the dataframe
* **Return type:**
  fig (go.Figure)

#### planet_yr_method_dist()

The planet year method distance visualization of Planet_Viz class.
* **Returns:**
  The planet year method distance figure of the dataframe
* **Return type:**
  fig (go.Figure)

#### planet_yr_method_hist(cumulative: bool = True)

The planet year method histogram visualization of Planet_Viz class.
* **Parameters:**
  **cumulative** (*bool*) – whether the histogram is cumulative or not, default is True
* **Returns:**
  The planet year method histogram figure of the dataframe
* **Return type:**
  fig (go.Figure)
