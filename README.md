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
Ensure that `pipeline_dist.joblib` and `pipeline_time.joblib` are downloaded in the same directory as the python files.

Then, go to the directory where `dashboard.py` and `predict.py` are installed, and run:

```
streamlit run dashboard.py
```
or:
```
streamlit run predict.py
```
They should open in the browser. If they do not, copy the IP address listed in the terminal.

For any other inevitable problems, consult the internet.

### File index

`dashboard.py` - A complete Streamlit application which allows users to train their own ML model on a dataset and see how it performs
`flowerdecisiontree.py` - Following the tutorial for `sklearn`'s decision tree models using the `iris` dataset
`folium.ipynb` - Following the tutorial for the `folium` plugin
`fortune500.csv` - A dataset for the Jupyter tutorial
`jupyter-tutorial.ipynb` - Following the tutorial for Jupyter
`map.png` and `nymap.png` and `taximap.png` - Supplementary to `taxidrives.ipynb`
`pipeline_dist.joblib` and `pipeline_time.joblib` - Outputs of `taximl.ipynb`, supplementary to `predict.py`
`predict.py` - A complete Streamlit application which allows a user to input parameters and see the prediction of the model
`taxidrives.ipynb` - **The biggest project**, a thorough and well-presented exploration of the taxi dataset in Jupyter
`taximl.ipynb` - The training of the models used in `predict.py`, and the sandbox for developing the code used in `dashboard.py`
