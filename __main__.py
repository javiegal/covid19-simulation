import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sklearn.impute import SimpleImputer


def exploratory_analysis(df, maps):
    """
    Performs an exploratory analysis over the COVID-19 simulation dataframe
    :param df: COVID-19 simulation dataframe
    :param maps: dictionaries for categorical values
    """
    fig = df.hist(column=['TEMP', 'HEART_RATE', 'SAT_O2', 'BLOOD_PRES_SYS', 'BLOOD_PRES_DIAS'], figsize=(15, 10),
                  xlabelsize=16, ylabelsize=16)
    [x.title.set_size(16) for x in fig.ravel()]
    plt.savefig("imgs/histogram.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='SEX', hue='EXITUS')

    ax.legend(maps['EXITUS'].values(), title='EXITUS')
    ax.set_xticklabels(maps['SEX'].values())
    plt.savefig("imgs/countplot.png")

    plt.figure(figsize=(8, 5))
    sns.displot(data=df, x='AGE', hue='EXITUS', kind='kde', fill=True, legend=False)
    plt.legend(maps['EXITUS'].values(), title='EXITUS')
    plt.savefig("imgs/displot_age.png")

    # convert the dataframe from wide to long form
    df_melt = df.drop(['SEX'], axis=1).melt(id_vars='EXITUS')
    plt.figure(figsize=(15, 10))
    sns.displot(data=df_melt, x='value', hue='EXITUS', kind='kde', fill=True, col='variable', legend=False)
    plt.xlim(1, 200)
    plt.ylim(0, 0.02)
    plt.legend(maps['EXITUS'].values(), title='EXITUS')
    plt.savefig("imgs/displot_multiple.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='EXITUS', y='AGE', data=df)
    ax.set_xticklabels(maps['EXITUS'].values())
    plt.savefig("imgs/boxplot_age.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='EXITUS', y='SAT_O2', data=df)
    ax.set_xticklabels(maps['EXITUS'].values())
    plt.savefig("imgs/boxplot_sat_o2.png")
    # Seems like there is a difference but these values are distorted
    # by the amount of zeros in SAT_02 (as expected lower values of SAT_O2
    # are normally related to EXITUS = YES)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='EXITUS', y='DAYS_HOSPITAL', data=df)
    ax.set_xticklabels(maps['EXITUS'].values())
    plt.savefig("imgs/boxplot_days_hospital.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='EXITUS', y='DAYS_ICU', data=df)
    ax.set_xticklabels(maps['EXITUS'].values())
    plt.savefig("imgs/boxplot_days_icu.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.pairplot(df, hue='EXITUS')
    ax.legend(maps['EXITUS'].values(), title='EXITUS')
    plt.savefig("imgs/pairplot.png")


def correlation_analysis(df):
    """
    Performs a correlation analysis over the COVID-19 simulation dataframe
    :param df: COVID-19 simulation dataframe
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.set(font_scale=1.5)
    hm = sns.heatmap(df.corr(), annot=True, ax=ax, vmin=-1, vmax=1, fmt='.2f', cmap='RdBu', cbar=False)
    hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize=15, rotation=60)
    hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize=15)

    plt.savefig("imgs/correlation_heatmap.png", transparent=True, bbox_inches='tight')


def survival_curves(df):
    """
    Calculates survival curves for the COVID-19 simulation dataframe
    :param df: COVID-19 simulation dataframe
    """
    # Survival curves
    time = df['DAYS_HOSPITAL']
    event_observed = df['EXITUS']

    # create a kmf object and fit it
    kmf = KaplanMeierFitter()
    kmf.fit(time, event_observed, label='Kaplan Meier Estimate')

    fig, ax = plt.subplots(figsize=(8, 5))

    # ci_show is meant for Confidence interval, since our data set is too tiny, thus we are not showing it.
    kmf.plot_survival_function(ci_show=False)

    ax.set_ylabel("est. probability of survival", fontsize=16)
    ax.set_xlabel("Days in hospital", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(prop={'size': 16})
    plt.savefig("imgs/kaplan_meier.png")

    # Separate ICU treatment and no treatment
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    g1 = df['DAYS_ICU'] == 0
    g2 = df['DAYS_ICU'] > 0
    kmf = KaplanMeierFitter()

    kmf.fit(time[g1], event_observed[g1], label='No ICU')  # fit the cohort 1 data
    kmf.plot_survival_function()

    kmf.fit(time[g2], event_observed[g2], label='ICU treatment')  # fit the cohort 2 data
    kmf.plot_survival_function()

    ax2.set_ylabel("est. probability of survival", fontsize=16)
    ax2.set_xlabel("Days in hospital", fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.legend(prop={'size': 16})
    plt.savefig("imgs/kaplan_meier_comp.png")


def preprocess(df):
    """
    Preprocessing of the COVID-19 simulation dataframe
    :param df: COVID-19 simulation dataframe
    :return: processed dataframe
    """
    print('--------------MISSING VALUES--------------')
    print(df.isnull().sum())

    print('\n--------------DATAFRAME SHAPE--------------')
    print(df.shape)

    # Remove 'DESTINATION' column and rows with a null value in the column 'EXITUS'
    df.pop('DESTINATION')
    df = df[df['EXITUS'].notna()]

    # Impute null values
    imp_cat = SimpleImputer(strategy='most_frequent')
    imp_num = SimpleImputer(strategy='median')

    df_num = df.select_dtypes(include=[np.number])
    df_cat = df.select_dtypes(exclude=[np.number])
    df_cat = pd.DataFrame(imp_cat.fit_transform(df_cat), columns=df_cat.columns, index=df_cat.index)
    df_num = pd.DataFrame(imp_num.fit_transform(df_num), columns=df_num.columns, index=df_num.index)
    df = pd.concat([df_cat, df_num], axis=1)

    pd.set_option('max_columns', None)
    print('\n--------------DATAFRAME DESCRIPTION--------------')
    print(df.describe())

    # Remove rows with errors in different columns
    print('\n--------------HIGHEST AGE. 10 ELEMENTS--------------')
    print(df.sort_values(by='AGE', ascending=False).head(10))
    df = df[df['AGE'] <= 120]

    print('\n--------------LESS DAYS IN HOSPITAL. 10 ELEMENTS--------------')
    print(df.sort_values(by='DAYS_HOSPITAL', ascending=True).head(10))
    df1 = df[(df.TEMP == 0) & (df.HEART_RATE == 0) & (df.GLUCOSE == 0) & (df.SAT_O2 == 0) & (df.BLOOD_PRES_SYS == 0) & (
            df.BLOOD_PRES_DIAS == 0)]

    print('\n--------------EXITUS INFORMATION ABOUT ELEMENTS WITH ALL DIAGNOSIS EQUAL TO ZERO--------------')
    print(df1.EXITUS.describe())
    df = df[(df.TEMP != 0) | (df.HEART_RATE != 0) | (df.GLUCOSE != 0) | (df.SAT_O2 != 0) | (df.BLOOD_PRES_SYS != 0) | (
            df.BLOOD_PRES_DIAS != 0)]

    print('\n--------------HIGHEST HEART RATE. 10 ELEMENTS--------------')
    print(df.sort_values(by='HEART_RATE', ascending=False).head(10))
    df = df[df['HEART_RATE'] <= 200]

    print('\n--------------HIGHEST BLOOD_PRES_SYS. 10 ELEMENTS--------------')
    print(df.sort_values(by='BLOOD_PRES_SYS', ascending=False).head(10))
    df = df[df['BLOOD_PRES_SYS'] <= 200]

    print('\n--------------HIGHEST_BLOOD_PRES_DIAS. 10 ELEMENTS--------------')
    print(df.sort_values(by='BLOOD_PRES_DIAS', ascending=False).head(10))
    df = df[df['BLOOD_PRES_DIAS'] <= 200]

    print('\n--------------NUMBER OF VALUES EQUAL TO ZERO--------------')
    print((df == 0).astype(int).sum(axis=0))
    print('\n--------------DATAFRAME SHAPE--------------')
    print(df.shape)

    df.pop('GLUCOSE')
    df['EXITUS'] = df['EXITUS'].astype('category')
    df['SEX'] = df['SEX'].astype('category')

    exitus_map = dict(enumerate(df['EXITUS'].cat.categories))
    sex_map = dict(enumerate(df['SEX'].cat.categories))

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    return df, {'EXITUS': exitus_map, 'SEX': sex_map}


def main():
    # Load and read dataset
    url = 'https://raw.githubusercontent.com/javiegal/covid19-simulation/main/COVID19_data.csv'
    initial_df = pd.read_csv(url, index_col=0)
    df, maps = preprocess(initial_df)

    # Save processed DataFrame
    df.to_csv('processed_COVID.csv', header=True, index=False)

    # exploratory_analysis(df, maps)
    # correlation_analysis(df)
    survival_curves(df)


if __name__ == "__main__":
    main()
