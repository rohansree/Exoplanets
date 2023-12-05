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
read_data(val = 1)
```
