# Study of Exoplanets: A Journey through Data & Space 🛰️🚀 
As part of the ECE 143 Fall 2023 final assignment, we have performed Exploratory Data Analysis (EDA) and Predictive Analysis on the [NASA Exoplanets Dataset](exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PSCompPars).<br>

We have attached an additional [documentation](documentation.md) for all the modules which we have developed for the parsing, visualising and training predictive models.

[Our Presentation](Study_of_Exoplanets_Group_18.pdf)

### Group Members

* Anand Kumar
* Atharva Yeola
* Colin Zhong
* Rohan Sreedhar
* Wang (Flannery) Liu

### Disclaimer ⚠️
The notebook can't show all the graphs in default github viewer. You can view all the images properly in [NBViewer](https://nbviewer.org/github/rohansree/Exoplanets/blob/main/Exoplanets_Viz.ipynb)
<br>

<b>Recommended</b>: Setup locally to view the interactive plotly plots.

## File Structure 📂

```
📦Exoplanets
 ┣ 📂data
 ┃ ┗ 📜NASA_planetary_data.csv
 ┣ 📂modules
 ┃ ┣ 📜discovery_viz.py
 ┃ ┣ 📜pred_model.py
 ┃ ┗ 📜planet_viz.py
 ┣ 📂plotly_plots
 ┃ ┗ Contains all the plotly images
 ┣ 📜documentation.md
 ┣ 📜Exoplanets_Viz.ipynb
 ┣ 📜Methods, Telescopes and Instruments.ipynb
 ┣ 📜README.md
 ┣ 📜requirements.txt
 ┗ 📜Study_of_Exoplanets_Group_18.pdf
 ```

## How to run 🧑‍💻
1. Clone the respository and run [`requirements.txt`](requirements.txt) file to install all the dependencies.
```
$ pip install -r requirements.txt
```
2. Run all the cells in the [Jupyter Notebook](Exoplanets_Viz.ipynb) to check out all the interactive plots.

## Third-Party Modules Used
All the modules are given in [`requirements.txt`](requirements.txt) and listed below:

```
geopy==2.4.1
matplotlib==3.7.1
numpy==1.24.2
pandas==2.0.0
plotly==5.18.0
scikit_learn==1.3.2
xgboost==2.0.2
```

## Acknowledgements 🙏
This research has made use of the NASA Exoplanet Archive, which is operated by the California Institute of Technology, under contract with the National Aeronautics and Space Administration under the Exoplanet Exploration Program.

## References 🗒️

1. <i>NASA exoplanet archive </i>. NASA Exoplanet Archive. (n.d.). https://exoplanetarchive.ipac.caltech.edu/ 



