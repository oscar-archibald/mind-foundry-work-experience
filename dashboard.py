import streamlit as st
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
import openml
from pathlib import Path
from joblib import dump, load

cache_file = Path(".openml.pkl")
if cache_file.exists():
    X, y = load(cache_file)
else:
    openml.datasets.list_datasets(output_format="dataframe")
    dataset = openml.datasets.get_dataset(43584)
    X, y, _, _ = dataset.get_data(dataset_format="dataframe")
    dump((X, y), cache_file)

col = list(X.columns)

st.title("Train your own model")

st.markdown(f"> This dataset contains information about {len(X)} taxi drives that took place in NYC in 2016.")
st.markdown("```df.head()```")

st.write(X.head())

st.write("### Choose the target variable:")

target = st.selectbox("Target", col, index=2)

col.remove(target)

st.write("### Choose your parameters to train the model from:")

# no_cols = st.slider("Number of parameters", 1, len(col), step=1, value=4)

cols = []

# for i in range(no_cols):
#     cols.append(st.selectbox(str(i+1), col))

lcol, mcol, rcol = st.columns(3)

count=0
for _col in col:
    if count%3==0:
        with lcol:
            if st.checkbox(_col):
                cols.append(_col)
    if count%3==1:
        with mcol:
            if st.checkbox(_col):
                cols.append(_col)
    if count%3==2:
        with rcol:
            if st.checkbox(_col):
                cols.append(_col)
    count+=1
    
if len(cols) == 0:
    st.error("Please select at least one parameter.")
    st.stop()

st.write("### Choose the test-size:")

split = st.slider("test-train split", 0.05, 0.95, step=0.05, value=0.9)

train_feat, test_feat, train_target, test_target = train_test_split(X[cols], X[target], test_size=split, random_state=1)

ohe = OneHotEncoder()
ohe.fit_transform(train_feat)

st.write("### Choose the model:")

models = {'Classifier': DecisionTreeClassifier, 'Regressor': DecisionTreeRegressor}
model_selected = st.selectbox("Options", models.keys())

transformer_tuples = []
for _col in cols:
    if X.dtypes[_col] == 'object':
        transformer_tuples.append((OneHotEncoder(), [_col]))

pipeline = make_pipeline(
    make_column_transformer(*transformer_tuples)
    , models[model_selected]())

pipeline.fit(train_feat, train_target)

score = pipeline.score(test_feat, test_target)

st.write("### $$R^2$$ score of your model")
st.write(score)
