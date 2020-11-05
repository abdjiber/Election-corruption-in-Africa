# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Author information
# %% [markdown]
# * Author: Djiberou Mahamadou Abdoul Jalil
# * Education: PhD student in computer science and Machine learning
# * Email: abdoul_jalil.djiberou_mahamadou@uca.fr
# * Copyrights: [Djiberou Mahamadou]('https://www.linkedin.com/pulse/imbalanced-datasets-resampling-techniques-abdoul-jalil/)
#
# %% [markdown]
# # Packages

# %%
import pandas as pd
import numpy as np
import warnings
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from plotly.offline import init_notebook_mode
from sklearn import preprocessing
from sklearn import decomposition
from IPython.display import Image

from utils import utils
import constants
import plotting
import scoring
import neural_nets
import machine_learning
import classification

init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')  # Ignoring the warnings

plt.style.use('dark_background')
params = {'legend.fontsize': 'large',
          'figure.figsize': (18, 5),
          'font.size': 12,
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          'axes.titlesize' :'large',
          'axes.grid': False
          }
rcParams.update(params)

np.random.seed(42)  # Setting a random seed for reproductibility

# %% [markdown]
# # Reading the data

# %%
df = pd.read_csv('../data/merged_r5_data.csv', low_memory=False)
print("Date shape {}".format(df.shape))

# %% [markdown]
# ## Data cleansing: Part 1
# %% [markdown]
# In this first part of the data cleansing we will drop unecessary variablesand those containg high percentage of missing data.

# %%
df = utils.clean_data(df.copy())

# %% [markdown]
# # Visualisation
# %% [markdown]
# We will visualize 4 variables distribution of the data as for:
# * Age: Q1
# * Countries: COUNTRY_ALPHA
# * Sex: Q101
# * Education level: Q97

# %%
ages = df.Q1
age_groups = utils.get_age_groups(ages)
countries = df.COUNTRY_ALPHA.map(constants.COUNTRIES_DICTIONARY)
sex = df.Q101.map(constants.SEX_DICTIONARY)
education = df.Q97.map(constants.EDUCATION_DICTIONARY)
age_groups_and_sex = [age_groups, sex]
age_groups_sex_labels = ['Age groups', 'Sex']
countries_label = ['Countries']
education_label = ['Education']

# %% [markdown]
# Please note that in the visualization figures the percentages are rounded, so the sum might not equal 100.

# %%
# Visualization of Country distribution
plotting.visualize([countries],
                   countries_label,
                   'Countries distribution',
                   direction='V'
                   )

# %% [markdown]
# From the figure above, one could note that there are two groups of countries that have approximately the same repartition. The one with a ratio of 5% (Kenya, South Africa, Ghana, Tanzania, Mozambique, Uganda, Nigeria and Malawi) and the other countries having a ratio of 2%.

# %%
# Visualization of Country age groups and sex distribution
plotting.visualize(age_groups_and_sex,
                age_groups_sex_labels,
                'Age groups, Sex distribution'
                )

# %% [markdown]
# From the figure above, one could see that the sex is equaly distributed contrary to the age groups in which the age group 25-35 is the most represented and >76 the least.

# %%
# Visualization of education level distribution
plotting.visualize([education], education_label, 'Education distribution')

# %% [markdown]
# 20% of people in the sample accomplished some secondary and high school and only 3% achieved a complete university curriculum.
# %% [markdown]
# ### Visualisation of most frequent types of bribes by country
# %% [markdown]
# Questions Q61A to Q61F refer to the types of bribes the interviewed persons experienced in the past. Here are the descriptions of each question:
# * Q61A: In the past year, how often, (if ever, have you had to pay a bribe, give a gift, or do a favor to
#        government officials in order to: Get a document or a permit)?
# * Q61B: In the past year, how often, if ever, have you had to pay a bribe, give a gift, or do a favor to
#     government officials in order to: Get water or sanitation services?
# * Q61C: In the past year, how often, if ever, have you had to pay a bribe, give a gift, or do a favor to
#     government officials in order to: Get treatment at a local health clinic or hospital?
# * Q61D: In the past year, how often, if ever, have you had to pay a bribe, give a gift, or do a favor to
#     government officials in order to: Avoid a problem with the police (like passing a checkpoint or avoiding a
#     fine or arrest)?
# * Q61E: In the past year, how often, if ever, have you had to pay a bribe, give a gift, or do a favor to
#     government officials in order to: Get a place in a primary school for a child?
# * Q61F: And during the last national election in 20xx, how often, if ever did a candidate or someone
#     from a political party offer you something, like food or a gift or money, in return for your vote?

# %%
"""Selecting variables Q61A to Q61F,
    replacing the missing data by the mode of each column,
    and encode the responses
"""
df_bribes = df.loc[:, 'Q61A':'Q61F'].copy()
df_bribes = (df_bribes.fillna(df_bribes.mode().loc[0])
                     .replace(constants.RESPONSES_EXPERIENCE_WITH_BRIBERIES,                                   value=None)
              )
df_bribes = df_bribes.replace({
                                'Once or twice': 'At least once',
                                'A few times': 'At least once',
                                'Often': 'At least once'
                                }, value=None
                               )


# %%
"""Getting a data frame containing the most
frequent type of bribe by country
"""
types = utils.get_most_corruption_type_by_country(df_bribes, countries)
df_most_corruption_type_by_country = pd.DataFrame(types)
encode_types = {'Avoid problem with police': 0,
                'Document or permit': 1,
                'Election incentives': 2,
                'Treatment at local health clinic or hospital': 3
                }
df_most_corruption_type_by_country.type.replace(encode_types, inplace=True)


# %%
# Uncomment to plot the choropleths
"""
utils.plot_choropleth(df=df_most_corruption_type_by_country,
                colorscale='RdBu',
                z=df_most_corruption_type_by_country.type,
                layout_title='Most frequent types of bribes in Africa',                               colorbar_title="Type of bribes",
                tickvals=list(encode_types.values()),
                ticktext =list(encode_types.keys())
                )

utils.plot_choropleth(df=df_most_corruption_type_by_country,
                z=df_most_corruption_type_by_country.percentage,
                layout_title='Bribes frequences',
                colorscale='Rainbow',
                colorbar_title="Frequence",
                tickmode='auto',
                tickvals=list(encode_types.values()),
                ticktext=list(encode_types.values())
                )
"""


# %%
get_ipython().run_cell_magic('html', '', "Please note that the two choropleth figures will not be displayed as output once the notebook in imported, so I'll put them manually.\n\n<img src='https://raw.githubusercontent.com/abdjiber/Election-corruption-in-Africa/master/img/figures/choropleth_types_bribes.png'>\n\n<img src='https://raw.githubusercontent.com/abdjiber/Election-corruption-in-Africa/master/img/figures/choropleth_types_bribes_frequencies.png.png'>")

# %% [markdown]
# ## data cleansing part 2 : Preparing data for modeling
# %% [markdown]
# Here we will set the target variable (Q61F), encode it to string for future plotting labels and drop it from the data frame. We will also fill the missing data by the mode of each variable.

# %%
target = df['Q61F']
df = df.fillna(df.mode().loc[0])  # Filling missing

"""
Getting a mask of target values different of -1,
998 and 9 which correspond respectively to missing,
reused to answer and don't know
"""
mask = (target != -1) & (target != 998) & (target != 9)
target = target[mask]
df = df.loc[target.index]
del df['Q61F']  # Deleting the target variable from the data.


# %%
target_2_classes = target.replace({0: 0, 1: 1, 2: 1, 3: 1})
target_2_classes.value_counts() / target.shape[0]


# %%
"""Scaling the data to the range [0, 1]"""
scale = preprocessing.MinMaxScaler()
scale.fit(df)
df_scaled = pd.DataFrame(scale.transform(df), columns=list(df))

# %% [markdown]
# # Modeling
# %% [markdown]
# ## Unsupervised learning
# %% [markdown]
# In the unsupervised learning part, we will be using principal components analysis (PCA), and K-means.
# %% [markdown]
# ### PCA

# %%
Pca = decomposition.PCA(n_components=2)
pca = Pca.fit_transform(df_scaled)
components = Pca.components_.T
columns = df.columns
explained_variance_ratio = Pca.explained_variance_ratio_.round(2)


# %%
plotting.plot_pca(pca, target_2_classes, explained_variance_ratio,
                  components, columns, figName='Principal components analysis')

# %% [markdown]
# ### K-means

# %%
from sklearn import cluster

# %% [markdown]
# The elbow criterion helps to determine the optimal number of clusters when using K-means.
# The process is as follow:
# 1. Set a list of number of clusters n1, n2, ...
# 2. Run the K-means using each n
# 3. For each n, compute the inertia (cost of the objective function)
# 4. Plot the list on inertia against the list of n
# 3. The optimal number of clusters will be the n giving an elbow.

# %%
plotting.elbow(pca)

# %% [markdown]
# From the figure above, the optimal number of clusters is 4. So we will run the K-means with 4 clusters.

# %%
kmeans = cluster.KMeans(n_jobs=-1, n_clusters=4)
pred_labs = kmeans.fit_predict(pca)


# %%
plotting.plot_2D(data=pca, labels=pred_labs, title='Data projection with K-means labels', components=components, columns=columns)

# %% [markdown]
# We can note that the clusters have different size. C2 has the biggest one, C4 and 3 have approximately the same size and
# C1 the smallest one. Once we obtained the clusters we can compare them on some variables. Here I choosed to do the comparision on:
# * Q1: Age
# * Q101: Sex
# * Q97: Education level
# * Q61A to Q61F: type of experienced corruption
# * COUNTRY_ALPHA: Countries

# %%
plotting.compare_clusters(df, pred_labs, {'Q101':'Cluster comparison on sex'})


# %%
plotting.compare_clusters(df, pred_labs, {'Q1': 'Cluster comparison on age'})


# %%
plotting.compare_clusters(df, pred_labs, {'Q97': 'Cluster comparison on education'})


# %%
plotting.compare_clusters(df, pred_labs, {'COUNTRY_ALPHA': 'Cluster comparison on countries'})


# %%
plotting.compare_clusters(target, pred_labs, {'Q61F':'Cluster comparison on receive bribe for election'})


# %%
plotting.compare_clusters(df, pred_labs, {'Q61A': 'Cluster comparison on pay                                         bribe for document or permit corruption'})


# %%
plotting.compare_clusters(df, pred_labs, {'Q61B': 'Cluster comparison on pay bribe for                                      water or sanitation services'})


# %%
plotting.compare_clusters(df, pred_labs, {'Q61C': 'Cluster comparision on pay                           bribe for treatment at local health clinic or hospital'})


# %%
plotting.compare_clusters(df, pred_labs, {'Q61D': 'Cluster comparision on pay                                               bribe to avoid problem with police'})


# %%
plotting.compare_clusters(df, pred_labs, {'Q61E': 'Cluster comparision on pay                                               bribe for school placement'})


# %%
plotting.plot_feature_importances(df, target_2_classes, pred_labs)

# %% [markdown]
# # Supervised learning
# %% [markdown]
# ## Importing packages for supervised learning

# %%
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# %% [markdown]
# ## Feature engineering
# %% [markdown]
# For the supervised learning, I used two common technics on categorical data:
# * Counting: Which consist of creating new features by counting the frequences of unique values of each feature.
# * Target encoding: Which consist of encoding the data using the target variable.

# %%
df_count_encode = utils.count_encoding(df.copy())
df_targ_encoded = utils.target_encode(df_count_encode, target_2_classes)


# %%
df_targ_encoded.head()

# %% [markdown]
# ## Splitting the data into train and validation sets

# %%
x_train, x_val, y_train, y_val = utils.split(df_targ_encoded, target_2_classes)


# %%
models_common_params_GBM = dict(random_state=constants.RANDOM_STATE, verbose=0)

# %% [markdown]
# ## Defining models
# %% [markdown]
# We will be using 6 models:
# * LightGBM
# * XGBoost
# * CatBoost
# * 2 layers Neural Nets
# * Explainable Boosting Classifier
# * Hist Gradient Boosting Classifier

# %%
clf_lgb = lgb.LGBMClassifier(**models_common_params_GBM)
clf_xgb = xgb.XGBClassifier(**models_common_params_GBM)
clf_cat = cat.CatBoostClassifier(**models_common_params_GBM)
nn = neural_nets.NeuralNets(x_train, x_val, y_train, y_val)
clf_int = ExplainableBoostingClassifier(random_state=constants.RANDOM_STATE)
clf_hist = HistGradientBoostingClassifier(random_state=constants.RANDOM_STATE)
list_models = {'LightGBM': clf_lgb, 'XGBoost': clf_xgb,
               'CatBoost': clf_cat, 'Neural Nets': nn,
               'Explainable Boosting': clf_int,
               'Hist Gradient boosting': clf_hist
               }


# %%
# Defining an instance of the classification class
classifiers = classification.Classification(list_models,
                                            x_train, x_val, y_train, y_val)

# %% [markdown]
# ## Supervised learning

# %%
df_scores_supLearning, prediction_supLearning = classifiers.run_supervised_learning()


# %%
df_scores_supLearning

# %% [markdown]
# ## Semi supervised learning

# %%
df_scores_supLearning, prediction_supLearning = classifiers.run_semi_supervised_learning()


# %%
df_scores_supLearning


# %%
model, oof,  = utils.CV(clf_lgb, df_targ_encoded, target_2_classes)


# %%
scoring.scores(target_2_classes,  np.where(oof < 0.5, 0, 1), oof)


