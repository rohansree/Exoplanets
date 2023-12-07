# modules package

## Submodules

## modules.discovery_viz module

### *class* modules.discovery_viz.Discovery_Viz(df_location: str, theme: str = 'plotly_dark', colors: list[str] = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'])

Bases: `object`

Class to visualize the discovery data

#### df

the dataframe to be visualized
* **Type:**
  pd.DataFrame

#### theme

the theme of the visualization, default is ‘plotly_dark’
* **Type:**
  str

#### colors

the list of colors for the visualization, default is [‘#636EFA’,  ‘#EF553B’,  ‘#00CC96’,  ‘#AB63FA’,  ‘#FFA15A’,  ‘#19D3F3’,  ‘#FF6692’,  ‘#B6E880’,  ‘#FF97FF’, ‘#FECB52’]
* **Type:**
  list[str]

### Examples

Sample usage of Discovery_Viz class:

```pycon
from modules.discovery_viz import Discovery_Viz
discovery_viz = Discovery_Viz('data/NASA_planetary_data.csv')
discovery_viz.plot_telescope_pie()
discovery_viz.ground_discovery_map()
discovery_viz.discovery_facility_hist()
discovery_viz.year_locale_scatter()
discovery_viz.locale_hist()
discovery_viz.locale_hist('telescope')
discovery_viz.locale_hist('instrument')
```

#### \_\_init_\_(df_location: str, theme: str = 'plotly_dark', colors: list[str] = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'])

The constructor for Discovery_Viz class.
* **Parameters:**
  * **df_location** (*str*) – the location of the dataframe to be visualized
  * **theme** (*str*) – the theme of the visualization, default is ‘plotly_dark’
  * **colors** (*list**[**str**]*) – the list of colors for the visualization, default is [‘#636EFA’,  ‘#EF553B’,  ‘#00CC96’,  ‘#AB63FA’,  ‘#FFA15A’,  ‘#19D3F3’,  ‘#FF6692’,  ‘#B6E880’,  ‘#FF97FF’, ‘#FECB52’]
* **Returns:**
  None

#### discovery_facility_hist(count_limit: int = 50)

Function to plot the histogram of the discovery facilities
* **Parameters:**
  **count_limit** (*int*) – the count limit of the discovery facilities, default is 50
* **Returns:**
  the histogram of the discovery facilities
* **Return type:**
  fig

#### *static* get_coordinates(location_name: str)

Function to get the coordinates of a location
* **Parameters:**
  **location_name** (*str*) – the name of the location
* **Returns:**
  the latitude and longitude of the location
* **Return type:**
  tuple

#### ground_discovery_map()

Function to plot the map of the discovery facilities on the ground
* **Returns:**
  the map of the discovery facilities on the ground
* **Return type:**
  fig

#### locale_hist(color_select: str = 'locale')

Function to plot the histogram of the discovery method and locale
* **Parameters:**
  **color_select** (*str*) – the color selection and it can have the values [‘locale’, ‘telescope’, ‘instrument’], default is ‘locale’
* **Returns:**
  the histogram of the discovery method and locale
* **Return type:**
  fig

#### plot_telescope_pie()

Function to plot the pie chart of the distribution of telescopes
* **Returns:**
  the pie chart of the distribution of telescopes
* **Return type:**
  plt

#### year_locale_scatter()

Function to plot the scatter plot of the year and locale of the discovery
* **Returns:**
  the scatter plot of the year and locale of the discovery
* **Return type:**
  fig

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

## modules.pred_model module

### *class* modules.pred_model.Model

Bases: `object`

This class contains the Linear Regression, Random Forest, and XGBoost models.

#### lr_model

Linear Regression model.
* **Type:**
  LinearRegression

#### rf_model

Random Forest model.
* **Type:**
  RandomForestRegressor

#### xgb_model

XGBoost model.
* **Type:**
  XGBRegressor

### Examples

```pycon
from pred_model import Model
model = Model()
lr_model, rf_model, xgb_model = model.train(df_x_train_norm, df_y_train_norm)
y_pred_lr, y_pred_rf, y_pred_xgb = model.predict(df_x_test_norm)
mse = model.calculate_mse(df_y_test_norm, (y_pred_lr + y_pred_rf + y_pred_xgb) / 3)
```

#### \_\_init_\_()

Initialize the Model class with Linear Regression, Random Forest, and XGBoost models.

#### calculate_mse(df_y_test_norm, y_pred_lr, y_pred_rf, y_pred_xgb)

Calculate the Mean Squared Error.
* **Parameters:**
  * **df_y_test_norm** (*pd.DataFrame*) – Normalized test target variable.
  * **y_pred_lr** (*np.ndarray*) – Predictions for Linear Regression model.
  * **y_pred_rf** (*np.ndarray*) – Predictions for Random Forest model.
  * **y_pred_xgb** (*np.ndarray*) – Predictions for XGBoost model.
* **Returns:**
  Mean Squared Error.
* **Return type:**
  float

#### predict(df_x_test_norm)

Make predictions using the trained models.
* **Parameters:**
  **df_x_test_norm** (*pd.DataFrame*) – Normalized test features.
* **Returns:**
  Predictions for Linear Regression, Random Forest, and XGBoost models.
* **Return type:**
  Tuple[np.ndarray, np.ndarray, np.ndarray]

#### train(df_x_train_norm, df_y_train_norm)

Train Linear Regression, Random Forest, and XGBoost models.
* **Parameters:**
  * **df_x_train_norm** (*pd.DataFrame*) – Normalized training features.
  * **df_y_train_norm** (*pd.DataFrame*) – Normalized training target variable.
* **Returns:**
  Trained models.
* **Return type:**
  Tuple[LinearRegression, RandomForestRegressor, XGBRegressor]

### *class* modules.pred_model.Visualizer(ft_ls, theme='dark')

Bases: `object`

This class contains the visualizations for the Mean Squared Error comparison, feature importance for Random Forest,
feature importance for XGBoost, and feature importance comparison between Random Forest and XGBoost.

#### ft_ls

List of feature names.
* **Type:**
  list

### Examples

```pycon
from pred_model import Visualizer
ft_ls = ['orbital period', 'orbital semi-major axis', 'orbital eccentricity',
                'radial velocity', 'transit depth', 'system rotational vel.']
visualizer = Visualizer(ft_ls)
visualizer.visualize_mse_comparison(mse_lr, mse_rf, mse_xgb)
visualizer.visualize_feature_importance_rf(feature_names_rf)
visualizer.visualize_feature_importance_xgb(feature_names_xgb)
visualizer.visualize_model_comparison(feature_names_rf, feature_names_xgb)
```

#### \_\_init_\_(ft_ls, theme='dark')

Initialize the Visualizer class with a list of feature names.
* **Parameters:**
  **ft_ls** (*list*) – List of feature names.

#### visualize_feature_importance_rf(feature_names_rf)

Visualize feature importance for Random Forest.
* **Parameters:**
  **feature_names_rf** (*np.ndarray*) – Feature importances for Random Forest.
* **Returns:**
  Plot of feature importance for Random Forest.
* **Return type:**
  plt

#### visualize_feature_importance_xgb(feature_names_xgb)

Visualize feature importance for XGBoost.
* **Parameters:**
  **feature_names_xgb** (*np.ndarray*) – Feature importances for XGBoost.
* **Returns:**
  Plot of feature importance for XGBoost.
* **Return type:**
  plt

#### visualize_model_comparison(feature_names_rf, feature_names_xgb)

Visualize feature importance comparison between Random Forest and XGBoost.
* **Parameters:**
  * **feature_names_rf** (*np.ndarray*) – Feature importances for Random Forest.
  * **feature_names_xgb** (*np.ndarray*) – Feature importances for XGBoost.
* **Returns:**
  Plot of feature importance comparison between Random Forest and XGBoost.
* **Return type:**
  plt

#### visualize_mse_comparison(mse_lr, mse_rf, mse_xgb)

Visualize Mean Squared Error comparison among different models.
* **Parameters:**
  * **mse_lr** (*float*) – Mean Squared Error for Linear Regression.
  * **mse_rf** (*float*) – Mean Squared Error for Random Forest.
  * **mse_xgb** (*float*) – Mean Squared Error for XGBoost.
* **Returns:**
  Plot of Mean Squared Error comparison among different models.
* **Return type:**
  plt
