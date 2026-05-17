import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import logging
from typing import List

# Configure logging and plot aesthetics
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
sns.set_theme(style="whitegrid", palette="muted")

class SubjectiveMetricsEDA:
    """
    A comprehensive suite for Exploratory Data Analysis on athlete monitoring data.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the EDA pipeline with the target DataFrame.
        """
        self.df = df.copy()
        # Identify numeric columns, excluding the dimensions (player_name, date)
        self.numeric_cols = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
        
    def analyze_missing_data(self) -> None:
        """Calculates and visualizes the percentage of missing values per column."""
        logging.info("Analyzing missing data...")
        missing_pct = self.df.isnull().mean() * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        if missing_pct.empty:
            logging.info("No missing data found.")
            return

        plt.figure(figsize=(10, 5))
        sns.barplot(x=missing_pct.values, y=missing_pct.index, palette="mako")
        plt.title("Percentage of Missing Data per Metric", fontsize=14)
        plt.xlabel("Missing Percentage (%)")
        plt.ylabel("Metrics")
        plt.tight_layout()
        plt.show()

    def plot_distributions_and_outliers(self) -> None:
        """
        Plots histograms for distributions and boxplots for univariate outlier detection.
        Iterates through all numeric columns.
        """
        logging.info("Plotting distributions and univariate outliers...")
        num_cols = len(self.numeric_cols)
        
        # Create a grid of subplots (2 columns wide)
        fig, axes = plt.subplots(nrows=(num_cols + 1) // 2, ncols=2, figsize=(15, 4 * ((num_cols + 1) // 2)))
        axes = axes.flatten()

        for i, col in enumerate(self.numeric_cols):
            # Plot Histogram with Kernel Density Estimate (KDE)
            sns.histplot(self.df[col].dropna(), kde=True, ax=axes[i], bins=30, color='steelblue')
            axes[i].set_title(f'Distribution of {col}', fontsize=12)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Frequency')

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()

        # Generate a separate consolidated view of boxplots for Outliers
        plt.figure(figsize=(15, 8))
        # Standardize data just for the boxplot so scales don't distort the visual
        df_standardized = (self.df[self.numeric_cols] - self.df[self.numeric_cols].mean()) / self.df[self.numeric_cols].std()
        sns.boxplot(data=df_standardized, orient="h", palette="Set2")
        plt.title("Univariate Outlier Detection (Data Standardized to Z-Scores)", fontsize=14)
        plt.xlabel("Z-Score (Standard Deviations from Mean)")
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self) -> None:
        """Computes and visualizes the Pearson correlation matrix for numeric features."""
        logging.info("Generating correlation matrix...")
        plt.figure(figsize=(12, 10))
        corr = self.df[self.numeric_cols].corr()
        
        # Mask the upper triangle for cleaner visualization
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                    vmax=1, vmin=-1, square=True, linewidths=.5, cbar_kws={"shrink": .75})
        plt.title("Feature Correlation Heatmap", fontsize=16)
        plt.tight_layout()
        plt.show()

    def detect_multivariate_anomalies(self) -> pd.DataFrame:
        """
        Uses an Isolation Forest to detect multivariate outliers (e.g., days where a player's 
        combination of Strain, Fatigue, and Sleep was highly abnormal).
        
        Returns:
            pd.DataFrame: A dataframe containing only the anomalous records.
        """
        logging.info("Running Isolation Forest for multivariate anomaly detection...")
        # Drop NaNs for ML modeling
        df_clean = self.df.dropna(subset=self.numeric_cols).copy()
        logging.info(f"Columns used for Isolation Forest {df_clean.columns}")
    

        if df_clean.empty:
            logging.warning("Not enough complete data (without NaNs) to run Isolation Forest.")
            return pd.DataFrame()

        # Initialize and fit the Isolation Forest model
        iso_forest = IsolationForest(contamination=0.03, random_state=42) # Assuming ~3% of days are true anomalies
        df_clean['anomaly_score'] = iso_forest.fit_predict(df_clean[self.numeric_cols])
        
        # Filter strictly for anomalies (-1 indicates anomaly)
        anomalies = df_clean[df_clean['anomaly_score'] == -1]
        logging.info(f"Detected {len(anomalies)} multivariate anomalies.")
        
        return anomalies

    def plot_longitudinal_team_trends(self, metrics_to_plot: List[str] = ['ATL', 'Strain']) -> None:
        """
        Plots the team's rolling average over time for specified metrics.
        
        Args:
            metrics_to_plot (List[str]): Columns to visualize longitudinally.
        """
        logging.info(f"Plotting longitudinal trends for {metrics_to_plot}...")
        # Ensure date is a datetime object
        if not np.issubdtype(self.df['date'].dtype, np.datetime64):
            self.df['date'] = pd.to_datetime(self.df['date'])

        # Aggregate to team average per day
        daily_avg = self.df.groupby('date')[metrics_to_plot].mean().reset_index()

        plt.figure(figsize=(15, 6))
        for metric in metrics_to_plot:
            # Apply a 7-day rolling average to smooth out daily noise
            sns.lineplot(
                x=daily_avg['date'], 
                y=daily_avg[metric].rolling(window=7, min_periods=1).mean(), 
                label=f'{metric} (7-Day Rolling Avg)', 
                linewidth=2
            )

        plt.title("Longitudinal Team Trends (Rolling Average)", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Metric Value", fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run_all(self):
        """Orchestrator to run the complete EDA suite."""
        print("Starting Sports Data EDA Suite...\n" + "-"*40)
        self.analyze_missing_data()
        self.plot_distributions_and_outliers()
        self.plot_correlation_matrix()
        
        # Time-series trend (Selecting a mix of load and wellness)
        cols_for_trends = [col for col in ['ATL', 'Readiness', 'Fatigue'] if col in self.numeric_cols]
        if cols_for_trends:
            self.plot_longitudinal_team_trends(metrics_to_plot=cols_for_trends)
            
        anomalies_df = self.detect_multivariate_anomalies()
        print("-" * 40 + "\nEDA Complete.")
        
        return anomalies_df