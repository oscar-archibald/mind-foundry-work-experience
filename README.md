# Mind Foundry Work Experience

This is a record of all the work that I did during my week of work experience at Mind Foundry.

### Things I learnt

* What a Jupyter notebook is and how to use it in the context of data science
* How to use a `pandas` dataframe
* Presenting data using `matplotlib.pyplot`
* Using `sklearn` to train and score ML models based on Decision Trees
* Building an interactive web app using `streamlit` to display data and predictions to users

### How to explore the repository

There are three 'finished' products:

`taxidrives.ipynb` is a complete and self-contained exploration of an `OpenML` dataset of taxi drives in NYC.

To be able to run the code:

```
pip install matplotlib
pip install openml
pip install folium
pip install pandas
pip install geopandas
```

#### Streamlit dashboards

To run this code:

```
pip install streamlit
pip install openml
pip install sklearn
pip install joblib
pip install pathlib
pip install streamlit-folium
```
[insert instructions for loading model]

Then, go to the directory where `dashboard.py` and `predict.py` are installed, and run :

```
streamlit run dashboard.py
```
or:
```
streamlit run predict.py
```
