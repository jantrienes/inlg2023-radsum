import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics
from sklearn.model_selection import learning_curve as learning_curve_


def label_distribution(y, out_file):
    fig, ax = plt.subplots()
    y.value_counts().sort_index().plot.bar(ax=ax)
    plt.xticks(rotation=0)
    plt.title("Label distribution")
    plt.xlabel("Label")
    plt.ylabel("Instances (N)")
    sns.despine()
    with PdfPages(out_file) as pdf:
        pdf.savefig(fig, bbox_inches="tight")


def classification_report(y_true, y_pred, out_txt, out_json):
    cr = metrics.classification_report(y_true, y_pred)
    print(cr)
    with open(out_txt, "w") as fout:
        fout.write(cr)

    cr = metrics.classification_report(y_true, y_pred, output_dict=True)
    with open(out_json, "w") as fout:
        fout.write(json.dumps(cr))


def confusion_matrix(pipe, X, y, out_file):
    fig, ax = plt.subplots()
    metrics.plot_confusion_matrix(
        pipe, X, y, normalize="true", cmap="Blues", values_format=".2f", ax=ax
    )
    with PdfPages(out_file) as pdf:
        pdf.savefig(fig, bbox_inches="tight")


def _plot_guidance_distribution(df, split, ax, legend=False):
    df_counts = pd.concat(
        [
            df["src"].apply(len).rename("Source Signal"),
            df["z"].apply(len).rename("Guidance Signal"),
        ],
        axis=1,
    )
    sns.histplot(df_counts, discrete=True, ax=ax, legend=legend)
    ax.set_title(split)
    ax.set_xlabel("Sentences (N)")
    ax.set_ylabel("Documents")


def guidance_distribution(df_train, df_valid, df_test, out_file):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)
    _plot_guidance_distribution(df_train, "train", axes[0])
    _plot_guidance_distribution(df_valid, "valid", axes[1])
    _plot_guidance_distribution(df_test, "test", axes[2], legend=True)
    plt.suptitle("Number of sentences per document", fontsize=16)
    sns.despine()

    with PdfPages(out_file) as pdf:
        pdf.savefig(fig, bbox_inches="tight")


def learning_curve(
    estimator,
    X,
    y,
    out_file=None,
    title="Learning Curve",
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    (
        train_sizes,
        train_scores,
        test_scores,
        fit_times,
        _,
    ) = learning_curve_(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    if out_file:
        with PdfPages(out_file) as pdf:
            pdf.savefig(fig, bbox_inches="tight")

    return fig
