import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans


# Create required folders
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# LOAD DATA
def load_data(path):
    try:
        df = pd.read_csv(path)
        print("Dataset loaded successfully.\n")
        return df

    except FileNotFoundError:
        print("File not found.")
        return None

    except Exception as e:
        print("Error:", e)
        return None


# CLEAN THE DATASET
def clean_data(df):

    print("\nCleaning dataset...")

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove null values
    df = df.dropna()

    numeric_cols = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    # Remove rows with invalid numeric data
    df = df.dropna()

    print("Cleaning complete.\n")

    return df


# Add unique filter logic
def apply_filter(df):

    filtered_df = df[df["humidity"] > 60]

    print(f"Filtered rows: {len(filtered_df)}")

    return filtered_df


# Feature Engineering
def feature_engineering(df):

    # NPK Ratio
    df["NPK_ratio"] = (
        df["N"] + df["P"] + df["K"]
    ) / 3

    # Nitrogen to Potassium Ratio
    df["N_to_K"] = df["N"] / df["K"]

    # Total Nutrients
    df["Total_Nutrients"] = (
        df["N"] + df["P"] + df["K"]
    )

    print("Feature engineering complete.\n")

    return df


# Outlier Detection
def remove_outliers(df):

    print("Removing outliers...\n")

    z_scores = np.abs(
        (df["N"] - df["N"].mean())
        / df["N"].std()
    )

    df = df[z_scores < 3]

    print("Outliers removed.\n")

    return df


# Analysis of data
def analyze_data(df):

    print("\nDescriptive Statistics")
    print(df.describe())

    print("\nVariance")
    print(df.var(numeric_only=True))

    print("\nStandard Deviation")
    print(df.std(numeric_only=True))

    print("\nMedian")
    print(df.median(numeric_only=True))

    print("\nCorrelation")

    correlation = df.corr(
        numeric_only=True
    )

    print(correlation)

    return correlation


# Clustering analysis
def clustering_analysis(df):

    print(
        "\nPerforming clustering analysis...\n"
    )

    X = df[["N", "P", "K"]]

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df["cluster"] = kmeans.fit_predict(X)

    print("Clustering complete.\n")

    return df


# Comparative analysis
def comparative_analysis(df):

    print("\nCLUSTER COMPARISON")

    comparison = df.groupby(
        "cluster"
    )[["N", "P", "K"]].mean()

    print(comparison)


# Static Visualizations
def visualize_data(df, correlation):

    # Scatter Plot
    plt.figure(figsize=(8, 6))

    for c in df["cluster"].unique():

        subset = df[
            df["cluster"] == c
        ]

        plt.scatter(
            subset["N"],
            subset["K"],
            label=f"Cluster {c}"
        )

    plt.xlabel("Nitrogen")
    plt.ylabel("Potassium")

    plt.title(
        "Clustered NPK Synergy"
    )

    plt.legend()

    plt.savefig(
        "outputs/scatter_plot.png"
    )

    plt.show()

    # Histogram
    plt.figure(figsize=(8, 6))

    plt.hist(df["N"], bins=20)

    plt.xlabel("Nitrogen")
    plt.ylabel("Frequency")

    plt.title(
        "Nitrogen Distribution"
    )

    plt.savefig(
        "outputs/histogram.png"
    )

    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))

    plt.imshow(correlation)

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns,
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        "outputs/heatmap.png"
    )

    plt.show()

    # Boxplot
    plt.figure(figsize=(8, 6))

    plt.boxplot(df["N"])

    plt.title("Nitrogen Boxplot")

    plt.savefig(
        "outputs/boxplot.png"
    )

    plt.show()


# Animated Visualization
def animated_graph(df):

    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):

        ax.clear()

        subset = df.iloc[:frame]

        ax.scatter(
            subset["N"],
            subset["K"]
        )

        ax.set_xlabel("Nitrogen")
        ax.set_ylabel("Potassium")

        ax.set_title(
            "Animated NPK Synergy Trend"
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=len(df),
        interval=50,
        repeat=False
    )

    # Save animation
    ani.save(
        "outputs/animated_scatter.gif",
        writer="pillow"
    )

    plt.show()

    return ani


# Animated Histogram
def animated_histogram(df):

    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):

        ax.clear()

        subset = df["N"].iloc[:frame]

        ax.hist(subset, bins=20)

        ax.set_title(
            "Animated Nitrogen Distribution"
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=len(df),
        interval=50,
        repeat=False
    )

    # Save animation
    ani.save(
        "outputs/animated_histogram.gif",
        writer="pillow"
    )

    plt.show()

    return ani


# Main Function
def main():

    df = load_data(
        "data/dataset_original.csv"
    )

    if df is None:
        return

    # Clean dataset
    df = clean_data(df)

    # Apply unique filter
    df = apply_filter(df)

    # Feature engineering
    df = feature_engineering(df)

    # Remove outliers
    df = remove_outliers(df)

    # Save cleaned dataset
    df.to_csv(
        "data/dataset_cleaned.csv",
        index=False
    )

    print("Cleaned dataset saved.\n")

    # Statistical Analysis
    correlation = analyze_data(df)

    # Clustering
    df = clustering_analysis(df)

    # Save clustered dataset
    df.to_csv(
        "data/dataset_clustered.csv",
        index=False
    )

    print("Clustered dataset saved.\n")

    # Comparative analysis
    comparative_analysis(df)

    # Visualization
    visualize_data(df, correlation)

    # Animation
    scatter_animation = animated_graph(df)
    hist_animation = animated_histogram(df)

    print("\nProject Complete.\n")


# Run program
if __name__ == "__main__":
    main()