import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.offline import iplot
from sklearn import cluster

from utils import utils
import constants


def visualize(list_cols_values, labels, figName, direction='H', saveFig=True):
    """Helper function for data visualisation"""
    fig = plt.figure(figsize=(12, 4))
    n_cols = len(list_cols_values)
    n_rows = 1
    for i, col_values in enumerate(list_cols_values):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        count_col_values = col_values.value_counts().sort_values()
        if direction == 'H':
            count_col_values.plot(kind='barh',
                                  ax=ax,
                                  label=labels[i],
                                  color=constants.COLORS[i])
        else:
            count_col_values.plot(kind='bar',
                                  ax=ax,
                                  label=labels[i],
                                  color=constants.COLORS[i])
        annotate(count_col_values, ax, direction)
    fig.tight_layout()
    fig.suptitle(figName, **constants.FIGURE_SUPTITLE_PARAMS)
    fig.legend()
    if saveFig:
        savefig(fig, figName)


def annotate(count_col_values, ax, direction):
    """Helper function for annotating figures"""
    counts_percentage = 100 * \
        count_col_values.apply(lambda x: x / count_col_values.sum())\
                        .round(2)
    for j in range(count_col_values.shape[0]):
        percentage = counts_percentage[j]
        if direction == 'H':
            ax.text(count_col_values[j], j, str(int(percentage)) + '%')
        else:
            ax.text(j,
                    count_col_values[j],
                    f'{int(percentage)} %',
                    weight='bold',
                    size=14)


def savefig(fig, name, path=constants.FIGURES_SAVING_PATH):
    """Helper function for saving figures"""
    full_path = f'{path}/{name}.png'
    fig.savefig(full_path, bbox_inches='tight', orientation='landscape')


def plot_choropleth(df, z, layout_title, colorscale, colorbar_title, tickvals,
                    ticktext):
    """Helper function to plot a choropleth of the most
       frequent types of bribes by country
    """
    tickmode = 'array'
    color = 'rgb(255,255,255)'

    data = [
        dict(type='choropleth',
             autocolorscale=False,
             locations=df.country,
             z=z,
             locationmode="country names",
             text=df.percentage,
             hoverinfo='location+text',
             colorscale=colorscale,
             marker=dict(line=dict(color=color, width=2)),
             colorbar=dict(title=colorbar_title,
                           tickmode=tickmode,
                           tickvals=tickvals,
                           ticktext=ticktext))
    ]
    layout = dict(title=layout_title,
                  geo=dict(scope='africa', projection=dict(type="mercator")))

    fig = dict(data=data, layout=layout)
    iplot(fig)


def plot_pca(data,
             labels,
             explained_variance,
             components,
             columns,
             figName,
             plot_correlation_circle_cyle=True,
             saveFig=True):
    """Helper function for plotting PCA"""
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 1, 1)
    colors = ['cyan', 'yellow', 'orange', 'darkgreen']
    for i, lab in enumerate(labels.unique()):
        ax1.scatter(data[labels == lab, 0],
                    data[labels == lab, 1],
                    label=lab,
                    color=colors[i],
                    s=60)
    ax1.set_xlabel(f'Explained variance {int(100*explained_variance[0])} %')
    ax1.set_ylabel(f'Explained variance {int(100*explained_variance[1])} %')
    ax1.axhline(y=0, color='grey')
    ax1.axvline(x=0, color='grey')
    ax1.set_title('Data projection')
    fig.suptitle(figName, **constants.FIGURE_SUPTITLE_PARAMS)
    fig.legend(loc='bottom')
    fig.tight_layout()
    savefig(fig, figName)


def plot_correlation_circle(components, columns, ax):
    """Helper function for plotting the
       correlation circle of the PCA results
    """
    ax.scatter(components[:, 0], components[:, 1], s=60, color='tomato')
    for i, _ in enumerate(columns):
        x = components[i, 0]
        y = components[i, 1]
        ax.text(x, y, columns[i])
    """
    The two lines code below which draw a circle are commented
    for figures clarity, please uncomment if you want to display the circle.
    """
    # t = np.linspace(0, np.pi*2,100)
    # ax.plot(np.cos(t), np.sin(t), linewidth=1) #draw a circle
    ax.axhline(y=0, color='grey')
    ax.axvline(x=0, color='grey')
    ax.set_title('Correlation circle')


def elbow(data):
    """Helper function fot the elbow experience"""
    n_clusters = [2, 3, 4, 5, 6, 7, 8]
    cost = []
    for n_cluster in n_clusters:
        kmeans = cluster.KMeans(n_jobs=-1, n_clusters=n_cluster)
        kmeans.fit_predict(data)
        cost.append(kmeans.inertia_)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(n_clusters, cost, marker='*')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')
    savefig(fig, 'Elbow')


def plot_2D(data,
            labels,
            components,
            columns,
            title,
            saveFig=True,
            legend=True):
    """Helper function for plotting a two dimensions data"""
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for lab in np.unique(labels):
        cluster_data = data[labels == lab, :]
        x = cluster_data[:, 0]
        y = cluster_data[:, 1]
        ax1.scatter(x, y, label=str(lab))
        ax1.text(x.mean(),
                 y.mean(),
                 f'C{lab + 1}',
                 weight='bold',
                 size=20,
                 color='black')
    plot_correlation_circle(components, columns, ax2)
    if legend:
        plt.legend().remove()
    fig.suptitle(title)
    if saveFig:
        savefig(fig, title)


def compare_clusters(df, k_means_labels, column, saveFig=True):
    """ Helper function for clusters comparision."""
    fig = plt.figure(figsize=(12, 6))
    col_key = list(column.keys())[0]
    col_val = list(column.values())[0]
    unique_labels = np.unique(k_means_labels)
    for i, lab in enumerate(unique_labels):
        ax = fig.add_subplot(2, 2, i + 1)
        if not isinstance(df, pd.core.series.Series):
            values = df.loc[k_means_labels == lab, col_key]
            # Creating a mask for dropping values 999, -1, 998.
            mask = (values != 999) & (values != -1) & (values != 998)
            values = values[mask]
        else:
            values = df[k_means_labels == lab]
        if col_key == 'Q1':
            values.plot(kind='box', ax=ax, color='orange')
            print('Mean age cluster {}: {}'.format(i + 1,
                                                   np.round(values.mean(), 2)))
        elif col_key == 'Q101':
            count = values.map(constants.SEX_DICTIONARY).value_counts()
            count.plot(kind='bar', ax=ax)
            plt.xticks(range(2), count.index, rotation='horizontal')
            annotate(count, ax, 'V')
        elif col_key == 'Q97':
            count = values.map(constants.EDUCATION_DICTIONARY).value_counts()
            count.plot(kind='barh', ax=ax)
            annotate(count, ax, 'H')
        elif col_key in ('Q61F', 'Q61A', 'Q61B', 'Q61C', 'Q61D', 'Q61E'):
            count = values.map(constants.RESPONSES_EXPERIENCE_WITH_BRIBERIES)\
                          .value_counts()
            count.plot(kind='barh', ax=ax)
            annotate(count, ax, 'H')
        elif col_key == 'COUNTRY_ALPHA':
            count = values.map(constants.COUNTRIES_DICTIONARY).value_counts()
            count.plot(kind='barh', ax=ax, figsize=(14, 14))
            annotate(count, ax, 'H')
        ax.set_title('Cluster {}'.format(i + 1))
    fig.tight_layout()
    fig.suptitle(col_val, **constants.FIGURE_SUPTITLE_PARAMS)
    if saveFig:
        savefig(fig, col_val)


def plot_feature_importances(df, target, kmeans_labels):
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('Clusters comparision based on feature importances',
                 **constants.FIGURE_SUPTITLE_PARAMS)
    for i, u in enumerate(np.unique(kmeans_labels)):
        mask = kmeans_labels == u
        df_cluster = df.loc[mask]
        target_ = target[mask]
        f_imp = utils.get_importante_variables(df_cluster, target_)
        ax = fig.add_subplot(2, 2, i + 1)
        f_imp.plot(kind='barh', ax=ax)
        ax.set_title('Cluster {}'.format(i + 1))
        ax.set_xlabel('Gini criterion')
    savefig(fig, 'Clusters comparision based on feature importances')
    fig.tight_layout()
