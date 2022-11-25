
# main librarys
import numpy as np
import pandas as pd
import seaborn as sns
import math

# visualization libraries
import matplotlib.pyplot as plt
import plotly.express as px

from wordcloud import WordCloud, STOPWORDS
from math import ceil
from pywaffle import Waffle

import plotnine
from plotnine import *

# inferential statistics
import statsmodels.stats.api as sms
import scipy.stats as stats

# modeling
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold,KFold , cross_validate, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, plot_confusion_matrix, f1_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVC, SVR

from optuna.integration import LightGBMPruningCallback

# DATA CLEANING


def copy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Takes pd.DataFrame as an input and returns it's copy"""
    return df.copy()


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Takes pd.DataFrame as an input and drops all columns that have more than 50 percent of NaN values.
    params: df: pd.DataFrame, which must be cleaned"""

    thresh = len(df) * 0.5
    df.dropna(axis=1, thresh=thresh, inplace=True)
    return df


def remove_extreme_outlier(df: pd.DataFrame) -> pd.DataFrame:
    """Takes pd.DataFrame as an input and removes all extreme outlies from columns.
    params: df:pd.DataFrame to clean."""

    Q1 = df.quantile(0.1)
    Q3 = df.quantile(0.9)
    IQR = Q3 - Q1

    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Takes to Data Frames and list of columns.
    Drops those columns
    Returns new dataframe
    """

    df = df.drop(columns=columns)
    return df


def drop_all_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Takes to Data Frames and drops all rows with NaN values."""

    df = df.dropna(axis=0)
    return df


def drop_nan_rows_from_certain_cols(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Takes pd.DataFrame and list of columns, drops rows with NaN values from those certain columns"""

    df.dropna(axis=0, subset=columns, inplace=True)
    return df


def insert_status(df: pd.DataFrame, number) -> pd.DataFrame:
    """Takes pd.DataFrame and inserts new column, named 'status' with certain set int value"""
    df["status"] = number
    return df


def insert_mean_of_two_columns(
    df: pd.DataFrame, col1: str, col2: str, new_col_name: str
) -> pd.DataFrame:
    """Takes pd.DataFrame and inserts new columns with values of the mean of specified two columns"""

    df[new_col_name] = df[[col1, col2]].mean(axis=1)
    return df


def year_month(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Takes pd.DataFrame as an input and return extracted year and month columns.
    
    Also converts year column into numeric dtype.
    params: df:pd.DataFrame to use:
            columns: str - name of the column, from which to extract year and month.
    """
    df[["month", "year"]] = df[column].str.split("-", 1, expand=True)
    df.year = pd.to_numeric(df.year)
    return df


def month_to_int(df, column) -> pd.DataFrame:
    """Changes the abbreviations of the month into numbers"""
    dict = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    df[column] = df[column].replace(dict)
    return df


def certain_column_lower(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Takes certain column of the pd.DataFrame and turns all values into upper case"""
    df[feature] = df[feature].str.lower()
    return df


def extract_year_month(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Takes pd.DataFrame column with date, turns it into datetime dtype and 
    extracts year and month into two new columns.
    params:df: pd.DataFrame tu use;
           columns: str - title of the column which has date
    returns: pd.DataFrame with new columns"""

    df[column] = pd.to_datetime(df[column])
    df["year"] = pd.DatetimeIndex(df[column]).year
    df["month"] = pd.DatetimeIndex(df[column]).month
    return df


def purpose_of_the_loan(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Takes pd.DataFrame with certain column and replaces purpose title, given by people, 
    into set category of loan purpose"""

    df[column] = df[column].str.lower()
    df[column] = df[column].astype(str)

    small_business = [
        "refinancing",
        "financing",
        "business",
        "small_business",
        "refinance",
        "starting",
        "project",
        "startup",
        "start-up",
        "company",
        "shop",
        "bussiness",
        "bussines",
        "production",
        "entrepreneur",
    ]

    debt_consolidation = [
        "debt!",
        "consolodation",
        "dept",
        "debts",
        "consolidating",
        "payoff",
        "consolidate",
        "debt_consolidation",
        "consolidation",
        "debt",
        "consolidated",
        "consilidation",
        "consolodate",
        "loan",
    ]

    major_purchase = [
        "buying",
        "purchase",
        "major_purchase",
        "expenses",
        "personal",
        "motorcycle",
        "life",
        "club",
        "buy",
        "dream",
        "christmas",
        "restaurant",
        "bike",
        "purchasing",
        "boat",
        "computer",
        "yamaha",
        "engine",
    ]

    credit_card = ["creditcard", "debit", "credit_card", "card", "credit"]

    home_improvement = [
        "home_improvement",
        "repairs",
        "improvements",
        "kitchen",
        "roof",
        "improve",
        "room",
        "renovation",
        "bathroom",
        "swimming",
        "basement",
        "fixing",
        "furniture",
        "building",
        "pool",
        "repair",
        "fix",
    ]

    house = [
        "land",
        "housing",
        "rental",
        "apartment",
        "estate",
        "property",
        "home",
        "rent",
        "studio",
    ]
    vacation = ["trip", "travel", "trailer", "vacation", "holiday"]

    medical = [
        "medical",
        "dental",
        "surgery",
        "nursing",
        "care",
        "rehab",
        "health",
        "hospital",
        "cancer",
    ]

    car = [
        "car",
        "truck",
        "track",
        "vehicle",
        "honda",
        "auto",
        "automobile",
        "bmw",
        "toyota",
        "ford",
    ]

    educational = [
        "teacher",
        "study",
        "learning",
        "university",
        "college",
        "education",
        "educational",
        "student",
        "school",
        "tuition",
        "degree",
        "course",
        "graduation",
        "academy",
    ]

    moving = ["moving", "move", "relocation", "transportation"]

    renewable_energy = [
        "energy",
        "solar",
        "green",
        "renewable_energy",
        "advisor",
        "insurance",
        "county",
        "legal",
        "adjuster",
        "billing",
        "law",
    ]

    wedding = [
        "married",
        "wife",
        "ring",
        "engagement",
        "divorce",
        "wedding",
        "county",
        "legal",
        "adjuster",
        "billing",
        "law",
        "marriage",
        "honeymoon",
    ]

    other = (
        wedding
        + renewable_energy
        + moving
        + educational
        + car
        + vacation
        + vacation
        + house
        + home_improvement
        + credit_card
        + major_purchase
        + debt_consolidation
        + small_business
    )

    df.loc[~df[column].str.contains("|".join(other)), column] = "other"
    df.loc[df[column].str.contains("|".join(wedding)), column] = "wedding"
    df.loc[
        df[column].str.contains("|".join(renewable_energy)), column
    ] = "renewable_energy"
    df.loc[df[column].str.contains("|".join(educational)), column] = "educational"
    df.loc[df[column].str.contains("|".join(moving)), column] = "moving"
    df.loc[df[column].str.contains("|".join(medical)), column] = "medical"
    df.loc[df[column].str.contains("|".join(car)), column] = "car"
    df.loc[df[column].str.contains("|".join(small_business)), column] = "small_business"
    df.loc[
        df[column].str.contains("|".join(debt_consolidation)), column
    ] = "debt_consolidation"
    df.loc[df[column].str.contains("|".join(major_purchase)), column] = "major_purchase"
    df.loc[df[column].str.contains("|".join(credit_card)), column] = "credit_card"
    df.loc[
        df[column].str.contains("|".join(home_improvement)), column
    ] = "home_improvement"
    df.loc[df[column].str.contains("|".join(house)), column] = "house"
    df.loc[df[column].str.contains("|".join(vacation)), column] = "vacation"
    return df


def rename_columns(df: pd.DataFrame, names: dict) -> pd.DataFrame:
    "Takes as an input pd.DataFrame and a dict with odl and new column names, changes them"
    df.rename(names, axis=1, inplace=True)
    return df


def change_value(
    df: pd.DataFrame, feature: str, odl_value: str, new_value: str
) -> pd.DataFrame:
    """Changes certain value in the pd.DataFrame column with new one"""
    df.loc[df[feature] == odl_value, feature] = new_value
    return df


def employment_type(df: pd.DataFrame, columns: str) -> pd.DataFrame:
    """Takes pd.DataFrame column as an input and replaces employment titles with categorized employment types"""

    df[columns] = df[columns].astype(str)

    business = [
        "manager",
        "sales",
        "business",
        "account",
        "consultant",
        "president",
        "management",
        "owner",
        "executive",
        "financial",
        "finance",
        "banker",
        "vp",
        "ceo",
        "dealer",
        "manger",
        "assistant",
        "finance",
        "cfo",
        "hr",
        "trainer",
        "bank",
        "planner",
        "recruiter",
        "director",
        "training",
        "agent",
        "instructor",
        "secretary",
        "buyer",
        "human",
        "realtor",
        "broker",
    ]

    tech_sector = [
        "analyst",
        "tech",
        "engineer",
        "technician",
        "senior",
        "lead",
        "support",
        "it",
        "developer",
        "server",
        "admin",
        "software",
        "programmer",
        "data",
        "processor",
        "system",
        "computer",
        "scientist",
        "technology",
        "network",
        "web",
        "tech.",
    ]

    government = [
        "federal",
        "government",
        "ministry",
        "committee",
        "chancellor",
        "specialist",
        "state",
        "institution",
        "secretary",
        "embassy",
        "administrate",
        "administrative",
        "administration",
        "administrator",
    ]

    education_culture_arts = [
        "school",
        "university",
        "education",
        "physician",
        "principal",
        "coach",
        "schools",
        "college",
        "prof",
        "painter",
        "dean",
        "professor",
        "artist",
        "art",
        "writer",
        "editor",
        "producer",
        "media",
        "design",
        "designer",
        "culture",
        "entertainment",
        "stylist",
        "teller",
        "teacher",
        "principal",
        "bookkeeper",
        "educator",
        "librarian",
        "caregiver",
        "examiner",
    ]

    health_medicine = [
        "nurse",
        "therapist",
        "medical",
        "health",
        "clinical",
        "hospital",
        "dental",
        "patient",
        "hygienist",
        "pharmacist",
        "pharmacy",
        "medic",
        "paramedic",
        "nursing",
        "healthcare",
        "pathologist",
        "care",
        "chemist",
    ]

    private_services = [
        "bartender",
        "truck",
        "maintenance",
        "electrician",
        "receptionist",
        "production",
        "hairdresser",
        "haircut",
        "machine",
        "controller",
        "carrier",
        "warehouse",
        "machinist",
        "bus",
        "cashier",
        "attendant",
        "dispatcher",
        "carpenter",
        "welder",
        "flight",
        "delivery",
        "shipping",
        "forklift",
        "journeyman",
        "transportation",
        "assembler",
        "assembly",
        "conductor",
        "crew",
        "courier",
        "electrical",
        "worker",
        "laborer",
        "handler",
        "logistics",
        "construction",
        "labor",
        "plumber",
        "electric",
        "air",
        "loader",
        "mechanic",
        "technologist",
        "architect",
        "mechanical",
    ]

    law = [
        "inspector",
        "counselor",
        "sergeant",
        "investigator",
        "firefighter",
        "sheriff",
        "fire",
        "federal",
        "associate",
        "detective",
        "deputy",
        "pilot",
        "captain",
        "forces",
        "marine",
        "army",
        "police",
        "cop",
        "policeman",
        "sherif",
        "atlaw",
        "judge",
        "officer",
        "representative",
        "attorney",
        "paralegal",
        "superintendent",
        "advisor",
        "insurance",
        "county",
        "legal",
        "adjuster",
        "billing",
        "law",
    ]

    all_words = (
        law
        + business
        + private_services
        + health_medicine
        + government
        + education_culture_arts
        + tech_sector
    )

    df.loc[~df[columns].str.contains("|".join(all_words)), columns] = "Other"
    df.loc[df[columns].str.contains("|".join(law)), columns] = "Law, enforcement"
    df.loc[df[columns].str.contains("|".join(business)), columns] = "Business"
    df.loc[
        df[columns].str.contains("|".join(private_services)), columns
    ] = "Private services, production, transportation"
    df.loc[
        df[columns].str.contains("|".join(education_culture_arts)), columns
    ] = "Education, arts, culture"
    df.loc[
        df[columns].str.contains("|".join(government)), columns
    ] = "Government sector"
    df.loc[df[columns].str.contains("|".join(tech_sector)), columns] = "IT, Tech sector"
    df.loc[
        df[columns].str.contains("|".join(health_medicine)), columns
    ] = "Medicine, pharmacy"

    return df


# Visualizations


def plot_total_percentage_of_loans(df: pd.DataFrame, column: str, title: str):
    """Takes certain pd.DataFrame column and plots waffle chart with percentage of different values
    params: df: pd.DataFrame to use;
            columns: str - name of the main plotted column;
            title: str - title of the waffle chart"""

    fig = plt.figure(
        figsize=(10, 10),
        FigureClass=Waffle,
        rows=5,
        values=df[column],
        colors=["green", "blue"],
        title={"label": title, "loc": "left"},
        icons="child",
        icon_size=30,
        icon_legend=True,
        labels=["{0} ({1:.2f}%)".format(k, v) for k, v in zip(df.index, df[column])],
        legend={
            "loc": "lower left",
            "bbox_to_anchor": (0, -0.4),
            "ncol": len(df),
            "framealpha": 0,
        },
    )
    fig.gca().set_facecolor("#EEEEEE")
    fig.set_facecolor("#EEEEEE")


def plot_box_stripplot(df: pd.DataFrame, x: str, y: str, title: str):
    """Takes as an input name of the pd.DataFrame, certain columns to plot on axis and plots boxplot+stripplot together.

    :param: df: the name of the pd.DataFrame to use;
            x: str - name of the column to plot on X axis;
            y: str - name of the column to plot on Y axis;
            title: str - title of the whole chart;
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_yscale("linear")
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    ax.set(xlabel="")
    sns.boxplot(x=x, y=y, data=df, palette="viridis")

    sns.stripplot(x=x, y=y, data=df, palette="GnBu", size=5, edgecolor="gray")

    sns.despine(trim=True, left=True)
    ax.set_title(title)


def plot_kde(df: pd.DataFrame, feature1: str, feature2: str):
    """Takes pd.Dataframe and two columns and plots their density plots (to compare) on one axis."""

    plt.figure(figsize=(15, 5))
    plt.title(f"KDE Plot: {feature1} vs. {feature2}", fontsize=30, fontweight="bold")
    sns.kdeplot(
        data=df,
        x=feature1,
        hue=feature2,
        fill=True,
        common_norm=False,
        palette="viridis",
        alpha=0.5,
        linewidth=0,
        legend=True,
    )
    sns.despine(right=True, left=True)
    # plt.xlim(300)
    plt.xlim(0)
    plt.tight_layout()


def plot_countplot(df: pd.DataFrame, feature1: str, feature2: str, title: str):
    """Takes as an input name of the pd.DataFrame and names of needed columns and plots a count plot.

    :param: df: the name of the pd.DataFrame to use;
            feature1: str - name of the columns to plot on X axis;
            feature2: str - name of the columns to plot as a 'hue' parameter (to do some cross counting);
            title: str -  final title (name) of the whole plot.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.countplot(
        y=feature1,
        hue=feature2,
        data=df,
        order=df[feature1].value_counts().index,
        palette="viridis",
    )

    if feature2 != None:
        ax.legend(loc="upper right", title=feature2)
        ax.bar_label(ax.containers[1])

    ax.set_title(title)
    ax.set(ylabel="")
    ax.set(xlabel=feature1)
    ax.bar_label(ax.containers[0])
    sns.despine(trim=True, left=True)
    plt.tight_layout()


def plot_boxplot(df: pd.DataFrame, x: str, y: str, title: str):
    """Takes as an input name of the pd.DataFrame, certain columns to plot on axis and plots boxplot.

    :param: df: the name of the pd.DataFrame to use;
            x: str - name of the column to plot on X axis;
            y: str - name of the column to plot on Y axis;
            title: str - title of the whole chart;
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_yscale("linear")
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    ax.set(xlabel="")
    sns.boxplot(x=x, y=y, data=df, palette="viridis")

    sns.despine(trim=True, left=True)
    ax.set_title(title)


def plot_barplot(df: pd.DataFrame, feature: str, title: str):
    """Takes as an input pd.DataFrame and it's column name and plots simple bar plot. 
    On x axis is plotted index of the given dataframe.
    params: df: pd.DataFrame to use;
            features: str - column to plot on y axis.
            title: str - name of the chart.
            """
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=df.index, y=df[feature], palette="viridis")
    plt.title(title)
    ax.bar_label(ax.containers[0])
    sns.despine(trim=True, left=True)


def plot_countplot_vertical(df: pd.DataFrame, feature1: str, feature2: str, title: str):
    """Takes as an input name of the pd.DataFrame and names of needed columns and plots a count plot.

    :param: df: the name of the pd.DataFrame to use;
            feature1: str - name of the columns to plot on X axis;
            feature2: str - name of the columns to plot as a 'hue' parameter (to do some cross counting);
            title: str -  final title (name) of the whole plot.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.countplot(
        y=feature1,
        hue=feature2,
        data=df,
        order=df[feature1].value_counts().index,
        palette="viridis",
    )

    if feature2 != None:
        ax.legend(loc="upper right", title=feature2)
        ax.bar_label(ax.containers[1])

    ax.set_title(title)
    ax.set(ylabel="")
    # ax.set(xlabel=feature1)
    ax.bar_label(ax.containers[0])
    sns.despine(trim=True, left=True)
    plt.tight_layout()


def multiple_violinplots(df: pd.DataFrame, feature1: str):
    """Plots multiple violin plots of 1 categorical feature and all other - numerical features.

    You need to make a new df only with needed columns: 1 categorical, other- numerical.
    params: df: pd.DatFrame, name of the data frame to use.
            feature1: str, categorical feature in which to compare numerical values of all other features.
    """
    plt.figure(figsize=(15, 10))
    for i in range(1, len(df.columns)):
        plt.subplot(int(len(df.columns) / 3) + 1, 2, i)
        ax = sns.violinplot(
            x=feature1, y=df.columns[i - 1], data=df, palette="viridis", dodge=True
        )
        ax.xaxis.grid(True)
        ax.set_title(f"{df.columns[i-1]}")
        ax.set(xlabel="")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set(ylabel="")
        sns.despine(right=True, left=True)
    plt.show()


def plot_line_plot_plotly(df: pd.DataFrame, x: str, y: str, z: str, title: str):
    """Takes as input name of the pd.DataFrame, names of columns ant plots a line plot.

    param: df:  the name of the pd.DataFrame to use;
            x: str - name of the column to plot on X axis;
            y: str - name of the column to plot on Y axis and to name the points on the line plot;
            z: str - name of the column to plot as hue (color) to the different lines;
            title: str - the title of the whole plot.
    """
    fig = px.line(df, x=x, y=y, color=z, text=y,)
    fig.update_traces(textposition="top left")
    fig.update_layout(legend_title="", title=title)
    fig.show()


def plot_countplot_horizontal(
    df: pd.DataFrame, feature1: str, feature2: str, title: str
):
    """Takes as an input name of the pd.DataFrame and names of needed columns and plots a count plot.

    :param: df: the name of the pd.DataFrame to use;
            feature1: str - name of the columns to plot on X axis;
            feature2: str - name of the columns to plot as a 'hue' parameter (to do some cross counting);
            title: str -  final title (name) of the whole plot.
    """

    fig, ax = plt.subplots(figsize=(15, 8))

    sns.countplot(
        x=feature1,
        hue=feature2,
        data=df,
        order=df[feature1].value_counts().index,
        palette="viridis",
    )

    if feature2 != None:
        ax.legend(loc="upper right", title=feature2)
        ax.bar_label(ax.containers[1])

    ax.set_title(title)
    ax.set(ylabel="")
    ax.set(xlabel=feature1)
    ax.bar_label(ax.containers[0])
    sns.despine(trim=True, left=True)
    plt.tight_layout()


def plot_world_cloud(df: pd.DataFrame, feature: str):
    """Plots the world cloud of all words, used in certain pd.DataFrame column."""
    all_words = ""
    stopwords = set(STOPWORDS)

    # iterate through the csv file
    for val in df[feature]:

        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        all_words += " ".join(tokens) + " "

    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color="black",
        stopwords=stopwords,
        min_font_size=10,
    ).generate(all_words)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()


def plot_grade_KDE(df: pd.DataFrame, feature: str):
    """Takes pd.DataFrame to use, numerical feature and plots it's densities in each grade group"""
    plt.figure(figsize=(15, 5))
    plt.title(f"KDE Plot: {feature} vs. grades", fontsize=30, fontweight="bold")
    ax = sns.kdeplot(df[df.grade == "A"][feature], label="grade 'A'", lw=2, legend=True)
    ax1 = sns.kdeplot(
        df[df.grade == "B"][feature], label="grade 'B'", lw=2, legend=True
    )
    ax2 = sns.kdeplot(
        df[df.grade == "C"][feature], label="grade 'C'", lw=2, legend=True
    )
    ax3 = sns.kdeplot(
        df[df.grade == "D"][feature], label="grade 'D'", lw=2, legend=True
    )
    ax4 = sns.kdeplot(
        df[df.grade == "E"][feature], label="grade 'E'", lw=2, legend=True
    )
    ax5 = sns.kdeplot(
        df[df.grade == "F"][feature], label="grade 'F'", lw=2, legend=True
    )
    ax5 = sns.kdeplot(
        df[df.grade == "G"][feature], label="grade 'G'", lw=2, legend=True
    )
    legend = ax.legend(loc="upper right")
    ax.yaxis.grid(True)
    sns.despine(right=True, left=True)
    plt.tight_layout()


def plot_categorical_countplots(df: pd.DataFrame, feature1: str) -> None:
    """ Takes as an input name of the pd.DataFrame (only categorical values) and one feature, which to use as a 'hue' factor.

    params: df: pd.DataFrame, consists only of categorical values in all columns.
            feature1: str - name of the last columns in the df, which will be used as a 'hue' factor in count plots.
    """
    plt.figure(figsize=(25, 35))
    for i in range(1, len(df.columns)):
        plt.subplot(int(len(df.columns)) + 1, 1, i)
        ax = sns.countplot(
            x=df.columns[i - 1], hue=feature1, data=df, palette="viridis"
        )
        ax.xaxis.grid(True)
        ax.set_xlabel("")
        ax.set_title(f"{df.columns[i-1]}")
        sns.despine(right=True, left=True)
    plt.show()


def plot_barplot_subgrade(df: pd.DataFrame, feature: str, title: str):
    """Takes as an input pd.DataFrame and it's column name and plots simple bar plot. 
    On x axis is plotted index of the given dataframe.
    params: df: pd.DataFrame to use;
            features: str - column to plot on y axis.
            title: str - name of the chart.
            """
    plt.figure(figsize=(20, 8))
    ax = sns.barplot(x=df.index, y=df[feature], palette="viridis")
    plt.title(title)
    ax.set(ylabel="percentage")
    ax.bar_label(ax.containers[0])
    sns.despine(trim=True, left=True)


# Inferential statistical analysis


class DiffMeans:
    """Module DiffMeans proceeds the needed table and all calculations
    for inferential statistical analysis of two the difference in means of to populations.
    Attribute:
    - df1;
    - df2;
    - feature1: first subgroup;
    - feature2: second subgroup;

    Methods of this class:
    - make_table;
    - diff_of_means;
    - sample_size_needed;
    - t_statistics (p_value);
    - conf_interval_of_difference"""

    def __init__(
        self, df1: pd.DataFrame, df2: pd.DataFrame, feature1: str, feature2: str
    ) -> None:
        self._df1 = df1
        self._df2 = df2
        self._feature1 = feature1
        self._feature2 = feature2

    def make_table(self) -> pd.DataFrame:
        """Creates a table - pd.DataFrame that helps to calculate the difference of means, estimated std."""
        self._table = pd.DataFrame(
            {
                "n": [len(self._df1), len(self._df2)],
                "mean": [
                    self._df1[self._feature1].mean(),
                    self._df2[self._feature2].mean(),
                ],
                "std": [
                    self._df1[self._feature1].std(),
                    self._df2[self._feature2].std(),
                ],
            },
            index=(["accepted", "rejected"]),
        )
        return self._table

    def diff_of_means(self) -> float:
        """Calculates the difference of two means."""
        self._diff = self._table.iloc[0]["mean"] - self._table.iloc[1]["mean"]
        return self._diff

    def sample_size_needed(self) -> None:
        """Calculates the required sample size to avoid p-hacking"""
        est_std = np.sqrt(
            (self._table.iloc[0]["std"] ** 2 + self._table.iloc[1]["std"] ** 2) / 2
        )
        effect_size = self._diff / est_std
        required_n = sms.NormalIndPower().solve_power(
            effect_size, power=0.8, alpha=0.05, ratio=1, alternative="larger"
        )
        required_n = ceil(required_n)
        print(f"Required sample size:{required_n}")

    def t_statistics(self) -> None:
        """Calculate the test statistic"""
        statistics, p_value = stats.ttest_ind(
            self._df1[self._feature1],
            self._df2[self._feature2],
            equal_var=False,
            alternative="greater",
        )
        print(f"T-statistic: {statistics}, p-value: {p_value}")

    def conf_interval_of_difference(self) -> None:
        """Calculates the confidence interval of the difference of two population means"""
        cm = sms.CompareMeans(
            sms.DescrStatsW(self._df1[self._feature1]),
            sms.DescrStatsW(self._df2[self._feature2]),
        )
        print(cm.tconfint_diff(usevar="unequal"))


if __name__ == "__main__":
    DiffMeans()


# Modeling


def log_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Takes as an input pd.DataFrame certain column and returns dataframe with new column of log_feature"""
    df["log_" + feature] = np.log(df[feature] + 0.0001)
    return df


def months_sin_cos(df: pd.DataFrame, feature: str):
    df["cos_" + feature] = np.cos(2 * math.pi * df[feature] / df[feature].max())
    df["sin_" + feature] = np.sin(2 * math.pi * df[feature] / df[feature].max())
    return df


def base_line(X: pd.DataFrame, y: pd.DataFrame, preprocessor: np.array) -> pd.DataFrame:
    """
    Takes as an input X (all usable predictors) and y (outcome, dependent variable) pd.DataFrames.
    The function performs cross validation with different already selected models.
    Returns metrics and results of the models in pd.DataFrame format.

    :param: X - pd.DataFrame of predictors(independent features);
            y - pd.DataFrame of the outcome;
            preprocessor: ColumnTransformer with all needed scalers, transformers;
    """

    balanced_accuracy = []
    roc_auc = []
    accuracy = []
    recall = []
    precision = []
    f1_score = []
    fit_time = []
    classifiers = ["Logistic regression", "SVC", "XGB classifier", "LGBM classifier"]

    models = [
        LogisticRegression(solver="saga", max_iter=1000, n_jobs=-1),
        SVC(),
        XGBClassifier(n_jobs=-1),
        LGBMClassifier(n_jobs=-1),
    ]

    for model in models:
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model),]
        )
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=3,
            scoring=(
                "balanced_accuracy",
                "accuracy",
                "f1_macro",
                "recall_macro",
                "precision_macro",
                "roc_auc",
            ),
        )
        fit_time.append(result["fit_time"].mean())
        balanced_accuracy.append(result["test_balanced_accuracy"].mean())
        accuracy.append(result["test_accuracy"].mean())
        recall.append(result["test_recall_macro"].mean())
        precision.append(result["test_precision_macro"].mean())
        f1_score.append(result["test_f1_macro"].mean())
        roc_auc.append(result["test_roc_auc"].mean())
    base_models = pd.DataFrame(
        {
            "Balanced accuracy": balanced_accuracy,
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "f1": f1_score,
            "Roc Auc": roc_auc,
            "Fit_time": fit_time,
        },
        index=classifiers,
    )
    base_models = base_models.style.background_gradient(cmap="Greens")
    return base_models


def plot_classifier_scores(
    model, X: pd.DataFrame, y: pd.DataFrame, predictions: np.array, target_labels: list
) -> None:
    """Plots the Confusion matrix and classification report from scikit-learn.
        
        :param: model - chosen model, modeled Pipeline from sklearn, on which data is trained.
                X - pd.DataFrame, X_train, X_validation, X_test data, which on to predict and plot the prediction 
                result.
                y - pd.DataFrame, the outcome, dependent variable: y_train. y_val, y_test, what to predict.
                predictions: y_hat, predictions from the model.
        """
    cmap = sns.dark_palette("seagreen", reverse=True, as_cmap=True)
    plot_confusion_matrix(
        model, X, y, normalize="true", cmap=cmap, display_labels=target_labels
    )
    plt.title("Confusion Matrix: ")
    plt.show()
    print(classification_report(y, predictions, target_names=target_labels))

    print()


def feature_names(
    module, numerical_features: list, binary_features: list, one_hot_features: list
) -> list:
    """
    Takes trained model.
    Extracts and returns feature name from preprocessor.
    """
    one_hot = list(
        module.named_steps["preprocessor"]
        .transformers_[1][1]
        .named_steps["one_hot_encoder"]
        .get_feature_names(one_hot_features)
    )

    binary = list(
        module.named_steps["preprocessor"]
        .transformers_[2][1]
        .named_steps["binary_encoder"]
        .get_feature_names()
    )
    cat_all = numerical_features + one_hot + binary

    return cat_all


def xgb_objective(trial, X: pd.DataFrame, y: pd.DataFrame, preprocessor) -> dict:
    """XGBoost hyper parameter searcher.

    Takes as an input X, y: pd.DataFrame and a Pipeline with 
    preprocessors, transformers and certain model, fits the given data and searches for the best 
    hyper parameters.

    :param: X: pd.DataFrame with features;
            y: pd.DataFrame with target (dependent variable);
            preprocessor: sklearn.Pipeline with all needed transformers, preprocessors;
    """

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 100),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 100),
        "gama": trial.suggest_float("gama", 0, 20),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 100),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    cv_scores = np.empty(5)

    for idx, (train_idx, eval_idx) in enumerate(cv.split(X, y)):
        X_train, X_eval = X.iloc[train_idx], X.iloc[eval_idx]
        y_train, y_eval = y.iloc[train_idx], y.iloc[eval_idx]

        X_train = preprocessor.fit_transform(X_train)
        X_eval = preprocessor.transform(X_eval)

        model = XGBClassifier(**param_grid, use_label_encoder=False)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            eval_metric="logloss",
            early_stopping_rounds=50,
            verbose=False,
        )
        preds = model.predict(X_eval)
        cv_scores[idx] = roc_auc_score(y_eval, preds)

    return np.mean(cv_scores)


def precision_recall_curve_opt_threshold(
    module, X_val: pd.DataFrame, y_val: pd.DataFrame
) -> None:
    """Takes as an input certain module:Pipeline and X_val, y_val sets and count the optimum 
    probability threshold from precision recall curve."""

    pred_prob = module.predict_proba(X_val)
    # Create the Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, pred_prob[:, 1])

    # Plot the ROC curve
    df_recall_precision = pd.DataFrame(
        {"Precision": precision[:-1], "Recall": recall[:-1], "Threshold": thresholds}
    )
    df_recall_precision.head()

    # Calculate the f-score
    fscore = (2 * precision * recall) / (precision + recall)

    # Find the optimal threshold
    index = np.argmax(fscore)
    thresholdOpt = round(thresholds[index], ndigits=4)
    fscoreOpt = round(fscore[index], ndigits=4)
    recallOpt = round(recall[index], ndigits=4)
    precisionOpt = round(precision[index], ndigits=4)

    # Create a data viz
    plotnine.options.figure_size = (8, 4.8)
    (
        ggplot(data=df_recall_precision)
        + geom_point(aes(x="Recall", y="Precision"), size=0.4)
        +
        # Best threshold
        geom_point(aes(x=recallOpt, y=precisionOpt), color="#981220", size=4)
        + geom_line(aes(x="Recall", y="Precision"))
        +
        # Annotate the text
        geom_text(
            aes(x=recallOpt, y=precisionOpt),
            label="Optimal threshold \n for class: {}".format(thresholdOpt),
            nudge_x=0.18,
            nudge_y=0,
            size=10,
            fontstyle="italic",
        )
        + labs(title="Recall Precision Curve")
        + xlab("Recall")
        + ylab("Precision")
        + theme_minimal()
    )
    print("Best Threshold: {} with F-Score: {}".format(thresholdOpt, fscoreOpt))
    print("Recall: {}, Precision: {}".format(recallOpt, precisionOpt))


def prob_trheshold_tuning(module: Pipeline, X_val: pd.DataFrame, y_val: pd.DataFrame) -> None:
    """Counts the optimal probability threshols step by step"""

    pred_prob = module.predict_proba(X_val)
    pred_prob = pred_prob[:, 1]

    # Array for finding the optimal threshold
    thresholds = np.arange(0.0, 1.0, 0.0001)
    fscore = np.zeros(shape=(len(thresholds)))
    print("Length of sequence: {}".format(len(thresholds)))

    # Fit the model
    for index, elem in enumerate(thresholds):
        # Corrected probabilities
        y_pred_prob = (pred_prob > elem).astype("int")
        # Calculate the f-score
        fscore[index] = f1_score(y_val, y_pred_prob)

    # Find the optimal threshold
    index = np.argmax(fscore)
    thresholdOpt = round(thresholds[index], ndigits=4)
    fscoreOpt = round(fscore[index], ndigits=4)
    print("Best Threshold: {} with F-Score: {}".format(thresholdOpt, fscoreOpt))

    # Plot the threshold tuning
    df_threshold_tuning = pd.DataFrame({"F-score": fscore, "Threshold": thresholds})
    df_threshold_tuning.head()

    plotnine.options.figure_size = (8, 4.8)
    (
        ggplot(data=df_threshold_tuning)
        + geom_point(aes(x="Threshold", y="F-score"), size=0.4)
        +
        # Best threshold
        geom_point(aes(x=thresholdOpt, y=fscoreOpt), color="#981220", size=4)
        + geom_line(aes(x="Threshold", y="F-score"))
        +
        # Annotate the text
        geom_text(
            aes(x=thresholdOpt, y=fscoreOpt),
            label="Optimal threshold \n for class: {}".format(thresholdOpt),
            nudge_x=0,
            nudge_y=-0.10,
            size=10,
            fontstyle="italic",
        )
        + labs(title="Threshold Tuning Curve")
        + xlab("Threshold")
        + ylab("F-score")
        + theme_minimal()
    )


def lgbm_objective(trial, X: pd.DataFrame, y: pd.DataFrame, preprocessor) -> float:
    """ Takes as an input pd.DataFrames with features and outcome, gives the best score of f1 after training and
        cross validation.

        :params: trial : a process of evaluating an objective function;
                X: pd.DataFrame with independent features (predictors);
                y: pd.DataFrame with the outcome (what to predict);
                numeric_features: list - names of the features, which must be scaled with scaler (numerical columns);
        :returns: the score, this time - f1 - after fitting data with different hyper parameters to the model and
        cross validation.
    """
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        model = LGBMClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            callbacks=[LightGBMPruningCallback(trial, "binary_logloss")],
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

        return np.mean(cv_scores)


def base_line_multi(
    X: pd.DataFrame, y: np.array, preprocessor: np.array
) -> pd.DataFrame:
    """
    Takes as an input X (all usable predictors) and y (outcome, dependent variable: multi-label) 
    pd.DataFrames.
    The the function performs cross validation with different already selected models.
    Returns metrics and results of the models in pd.DataFrame format.

    :param: X - pd.DataFrame of predictors(independent features);
            y - pd.DataFrame of the outcome;
            preprocessor: ColumnTransformer with all needed scalers, transformers;
    """
    balanced_accuracy = []
    accuracy = []
    recall = []
    precision = []
    f1_score = []
    fit_time = []
    classifiers = ["Logistic regression", "SVC", "XGB classifier", "LGBM classifier"]

    models = [
        LogisticRegression(
            solver="saga", multi_class="multinomial", max_iter=1000, n_jobs=-1
        ),
        SVC(),
        XGBClassifier(objective="multi:softmax", n_jobs=-1),
        LGBMClassifier(objective="multiclass", n_jobs=-1),
    ]

    for model in models:
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model),]
        )
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=3,
            scoring=(
                "balanced_accuracy",
                "accuracy",
                "f1_macro",
                "recall_macro",
                "precision_macro",
            ),
        )
        fit_time.append(result["fit_time"].mean())
        balanced_accuracy.append(result["test_balanced_accuracy"].mean())
        accuracy.append(result["test_accuracy"].mean())
        recall.append(result["test_recall_macro"].mean())
        precision.append(result["test_precision_macro"].mean())
        f1_score.append(result["test_f1_macro"].mean())
    base_models = pd.DataFrame(
        {
            "Balanced accuracy": balanced_accuracy,
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "f1": f1_score,
            "Fit_time": fit_time,
        },
        index=classifiers,
    )
    base_models = base_models.style.background_gradient(cmap="Greens")
    return base_models


def SVC_objective(trial, X: pd.DataFrame, y: pd.DataFrame, preprocessor) -> float:
    """Takes as an input pd.DataFrames with features and outcome, gives the best score of f1 after training and
    cross validation.

    :params: trial : a process of evaluating an objective function;
            X: pd.DataFrame with independent features (predictors);
            y: pd.DataFrame with the outcome (what to predict);
            preprocessor;
    :returns: the score, this time - f1 - after fitting data with different hyper parameters to the model and
    cross validation.
    """

    # (a) List all dimensionality reduction options
    dim_red = trial.suggest_categorical("dim_red", ["PCA", None])

    # (b) Define the PCA algorithm and its hyperparameters
    if dim_red == "PCA":
        pca_n_components = trial.suggest_int("pca_n_components", 2, 35)
        dimen_red_algorithm = PCA(n_components=pca_n_components)
    # (c) No dimensionality reduction option
    else:
        dimen_red_algorithm = "passthrough"

    # -- Instantiate estimator model
    svc_C = trial.suggest_loguniform("svc_C", 1e-5, 100)
    svc_kernel = trial.suggest_categorical("svc_kernel", ["rbf", "poly", "linear"])
    svc_gamma = trial.suggest_float("svc_gamma", 0.01, 1.0)
    svc_class_weight = trial.suggest_categorical("svc_class_weight", ["balanced", None])

    estimator = SVC(
        C=svc_C, kernel=svc_kernel, gamma=svc_gamma, class_weight=svc_class_weight
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("reduction", dimen_red_algorithm),
            ("estimator", estimator),
        ]
    )

    # -- Evaluate the score by cross-validation
    score = cross_val_score(pipeline, X, y, scoring="f1_macro")
    f1 = score.mean()
    return f1


def feature_names_num_hot_bin_ord(
    module,
    numerical_features: list,
    binary_features: list,
    one_hot_features: list,
    ordinal_features: list,
) -> list:
    """
    Takes trained model.
    Extracts and returns feature names from preprocessor.
    """
    one_hot = list(
        module.named_steps["preprocessor"]
        .transformers_[1][1]
        .named_steps["one_hot_encoder"]
        .get_feature_names(one_hot_features)
    )

    binary = list(
        module.named_steps["preprocessor"]
        .transformers_[2][1]
        .named_steps["binary_encoder"]
        .get_feature_names()
    )
    cat_all = numerical_features + one_hot + binary + ordinal_features

    return cat_all


def xgb_objective_multi(trial, X: pd.DataFrame, y: pd.DataFrame, preprocessor) -> dict:
    """XGBoost hyper parameter searcher.

    Takes as an input X, y: pd.DataFrame and a Pipeline with 
    preprocessors, transformers and certain model, fits the given data and searches for the best 
    hyper parameters.

    :param: X: pd.DataFrame with features;
            y: pd.DataFrame with target (dependent variable);
            preprocessor: sklearn.Pipeline with all needed transformers, preprocessors;
    """

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 100),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 100),
    }

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=123)
    cv_scores = np.empty(2)

    for idx, (train_idx, eval_idx) in enumerate(cv.split(X, y)):
        X_train, X_eval = X.iloc[train_idx], X.iloc[eval_idx]
        y_train, y_eval = y.iloc[train_idx], y.iloc[eval_idx]

        X_train = preprocessor.fit_transform(X_train)
        X_eval = preprocessor.transform(X_eval)

        model = XGBClassifier(
            objective="multi:softmax", **param_grid, use_label_encoder=False
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            eval_metric="mlogloss",
            early_stopping_rounds=50,
            verbose=False,
        )
        preds = model.predict(X_eval)
        cv_scores[idx] = f1_score(y_eval, preds, average="macro")

    return np.mean(cv_scores)


def xgb_objective_multi(trial, X: pd.DataFrame, y: pd.DataFrame, preprocessor) -> dict:
    """XGBoost hyper parameter searcher.

    Takes as an input X, y: pd.DataFrame and a Pipeline with 
    preprocessors, transformers and certain model, fits the given data and searches for the best 
    hyper parameters.

    :param: X: pd.DataFrame with features;
            y: pd.DataFrame with target (dependent variable);
            preprocessor: sklearn.Pipeline with all needed transformers, preprocessors;
    """

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 100),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 100),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 100),
    }

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=123)
    cv_scores = np.empty(2)

    for idx, (train_idx, eval_idx) in enumerate(cv.split(X, y)):
        X_train, X_eval = X.iloc[train_idx], X.iloc[eval_idx]
        y_train, y_eval = y.iloc[train_idx], y.iloc[eval_idx]

        X_train = preprocessor.fit_transform(X_train)
        X_eval = preprocessor.transform(X_eval)

        model = XGBClassifier(
            objective="multi:softmax", **param_grid, use_label_encoder=False
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            eval_metric="mlogloss",
            early_stopping_rounds=50,
            verbose=False,
        )
        preds = model.predict(X_eval)
        cv_scores[idx] = f1_score(y_eval, preds, average="macro")

    return np.mean(cv_scores)


def train_xgb_subgrade_default(
    grade: str,
    df_training: pd.DataFrame,
    df_validation: pd.DataFrame,
    preprocessor: Pipeline,
) -> tuple:
    """Takes ne name of the chosen grade, pre processes the given training and validation sets, trains XGBClassifier.
    
    params: grade: str - 'A', 'B', 'C', 'D', 'E', 'F', 'G';
            df_training:pd.DataFrame, 
            df_validation:pd.DataFrame, 
            preprocessor: final Columns transformer with all needed transformations;
    returns: pd.DataFrame of the metrics"""

    per_grade_training = df_training[df_training["grade"] == grade]
    per_grade_validation = df_validation[df_validation["grade"] == grade]
    y_train = per_grade_training["sub_grade"]
    y_test = per_grade_validation["sub_grade"]
    X_train = per_grade_training.drop(
        columns=[
            "grade",
            "sub_grade",
            "int_rate",
            "status",
            "cos_month",
            "sin_month",
            "log_loan_amnt",
            "log_annual_inc",
            "log_avg_cur_bal",
            "log_dti",
            "log_fico_score",
            "log_last_fico_score",
        ]
    )
    X_test = per_grade_validation.drop(
        columns=[
            "sub_grade",
            "grade",
            "int_rate",
            "status",
            "cos_month",
            "sin_month",
            "log_loan_amnt",
            "log_annual_inc",
            "log_avg_cur_bal",
            "log_dti",
            "log_fico_score",
            "log_last_fico_score",
        ]
    )

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(verbosity=0, random_state=123, use_label_encoder=False),
            ),
        ]
    ).fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metric_dict = pd.DataFrame(
        {
            "Grade": grade,
            "F1-score": f1_score(y_test, y_pred, average="macro"),
            "Precision": precision_score(y_test, y_pred, average="macro"),
            "Recall": recall_score(y_test, y_pred, average="macro"),
        },
        index=[str(grade) + "1 - " + str(grade) + "5"],
    )

    return metric_dict, model


def base_line_regression(
    X: pd.DataFrame, y: pd.DataFrame, preprocessor: ColumnTransformer
) -> pd.DataFrame:
    """
    Takes as an input X (all usable predictors) and y (outcome, dependent variable) pd.DataFrames.
    The the function performs cross validation with different already selected models.
    Returns metrics and results of the models in pd.DataFrame format.

    :param: X - pd.DataFrame of predictors(independent features);
            y - pd.DataFrame of the outcome;
            preprocessor: ColumnTransformer with all needed scalers, transformers.
    """
    mae = []
    mse = []
    rmse = []
    r2 = []
    fit_time = []
    regressors = ["Elastic Net", "SVR", "XGB Regressor", "LGBM Regressor"]

    models = [
        ElasticNet(),
        SVR(),
        XGBRegressor(),
        LGBMRegressor(),
    ]
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    for model in models:
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", model),]
        )
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=(
                "neg_mean_absolute_error",
                "neg_mean_squared_error",
                "neg_root_mean_squared_error",
                "r2",
            ),
        )
        fit_time.append(result["fit_time"].mean())
        mae.append(result["test_neg_mean_absolute_error"].mean() * -1)
        mse.append(result["test_neg_mean_squared_error"].mean() * -1)
        rmse.append(result["test_neg_root_mean_squared_error"].mean() * -1)
        r2.append(result["test_r2"].mean())
    base_models = pd.DataFrame(
        {
            "Mean_absolute_error": mae,
            "Mean_squared_error": mse,
            "Root_mean_squared_error": rmse,
            "R2": r2,
            "Fit time": fit_time,
        },
        index=regressors,
    )
    return base_models


def LGBM_regressor_objective(
    trial, X: pd.DataFrame, y: pd.DataFrame, preprocessor: ColumnTransformer
) -> float:
    """Takes as an input pd.DataFrames with features and outcome, preprocessor, gives the best scores after training and
        cross validation.

        :params: trial : a process of evaluating an objective function;
                X: pd.DataFrame with independent features (predictors);
                y: pd.DataFrame with the outcome (what to predict);
                preprocessor: sklearn ColumnsTransformer with all needed preprocessors.
        :returns: the score: neg_root_mean_squared_error, multiplied by -1
    """

    param = {
        "metric": "rmse",
        "random_state": 123,
        "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        "subsample": trial.suggest_categorical(
            "subsample", [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]
        ),
        "max_depth": trial.suggest_categorical("max_depth", [10, 20, 100]),
        "num_leaves": trial.suggest_int("num_leaves", 1, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 300),
        "cat_smooth": trial.suggest_int("min_data_per_groups", 1, 100),
    }

    cv = KFold(n_splits=2, random_state=123, shuffle=True)

    model = LGBMRegressor(**param)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", model),])

    score = cross_val_score(
        pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
    ).mean()

    return score * -1