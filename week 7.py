import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

plt.style.use('seaborn')

def load_and_explore_data():
    try:
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        print("First few rows of the dataset:")
        print(df.head())
        print("\nDataset Info:")
        print(df.info())
        print("\nMissing values:")
        print(df.isnull().sum())
        df_with_na = df.copy()
        df_with_na.loc[:2, 'sepal length (cm)'] = np.nan
        df_cleaned = df_with_na.fillna(df_with_na.mean(numeric_only=True))
        print("\nAfter cleaning missing values:")
        print(df_cleaned.head())
        return df_cleaned
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def analyze_data(df):
    try:
        print("\nBasic Statistics:")
        print(df.describe())
        grouped = df.groupby('species').mean(numeric_only=True)
        print("\nMean values by species:")
        print(grouped)
        print("\nObservations:")
        print("1. Different species show varying measurements.")
        print("2. Setosa tends to have smaller measurements overall.")
        print("3. Virginica generally has larger petal measurements.")
        return grouped
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return None

def create_visualizations(df):
    try:
        plt.figure(figsize=(10, 6))
        for species in df['species'].unique():
            species_data = df[df['species'] == species]
            plt.plot(species_data.index, species_data['sepal length (cm)'], 
                    label=species)
        plt.title('Sepal Length Trend by Species')
        plt.xlabel('Sample Index')
        plt.ylabel('Sepal Length (cm)')
        plt.legend()
        plt.show()
        grouped = df.groupby('species').mean(numeric_only=True)
        plt.figure(figsize=(10, 6))
        grouped['petal length (cm)'].plot(kind='bar')
        plt.title('Average Petal Length by Species')
        plt.xlabel('Species')
        plt.ylabel('Average Petal Length (cm)')
        plt.xticks(rotation=45)
        plt.show()
        plt.figure(figsize=(10, 6))
        df['sepal width (cm)'].hist(bins=20)
        plt.title('Distribution of Sepal Width')
        plt.xlabel('Sepal Width (cm)')
        plt.ylabel('Frequency')
        plt.show()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', 
                       hue='species', size='petal width (cm)', legend='full')
        plt.title('Sepal Length vs Petal Length by Species')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend(title='Species')
        plt.show()
    except Exception as e:
        print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    df = load_and_explore_data()
    if df is not None:
        grouped_data = analyze_data(df)
        create_visualizations(df)
