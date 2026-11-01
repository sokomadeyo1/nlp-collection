__all__ = ["flatten", "plot_dist"]

import seaborn as sns
from matplotlib.pyplot import show

sns.set_style("whitegrid")


def flatten(lists):
    return [item for list in lists for item in list]


def plot_dist(dist):
    sns.barplot(dict(dist))
    show()
