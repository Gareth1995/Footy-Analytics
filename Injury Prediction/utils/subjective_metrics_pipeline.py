import pandas as pd
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class PlayerMetricsPipeline:
    def __init__(self, mapping):
        self.mapping = mapping
        self.target_columns = [
            'player_name', 'date', 'ATL', 'Weekly load', 'Monotony', 'Strain', 
            'ACWR', 'CTL28', 'CTL42', 'Fatigue', 'Mood', 'Readiness', 
            'Sleep duration', 'Soreness', 'Stress'
        ]

    def _melt_file(self, file_path, metric_name):
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"  ⚠️ Skipped: {file_path} is completely empty.")
                return pd.DataFrame()

            # The first column is our date column
            date_col = df.columns[0]
            
            # Melt wide to long
            melted = df.melt(id_vars=[date_col], var_name='player_name', value_name=metric_name)
            melted = melted.rename(columns={date_col: 'date'})
            
            # STRICT Date Parsing: Explicitly tell Pandas it is DD.MM.YYYY
            melted['date'] = pd.to_datetime(melted['date'], format='%d.%m.%Y', errors='coerce')
            
            # Diagnostic: Did the dates parse correctly?
            valid_dates = melted['date'].notna().sum()
            if valid_dates == 0:
                print(f"  🚨 ERROR: Date parsing failed for {file_path}. All dates became NaT.")
                return pd.DataFrame()
            
            # Clean up trailing empty rows
            clean_df = melted.dropna(subset=['date'])
            print(f"  ✅ Processed {metric_name}: Generated {len(clean_df)} rows.")
            return clean_df

        except Exception as e:
            print(f"  ❌ ERROR processing {file_path}: {e}")
            return pd.DataFrame()

    def run(self):
        print("🚀 Starting pipeline execution...\n")
        merged_df = None
        
        for file_path, metric_name in self.mapping.items():
            df_metric = self._melt_file(file_path, metric_name)
            
            if df_metric.empty:
                continue
                
            if merged_df is None:
                merged_df = df_metric
            else:
                merged_df = pd.merge(merged_df, df_metric, on=['date', 'player_name'], how='outer')
        
        if merged_df is None or merged_df.empty:
            print("\n🚨 CRITICAL: Pipeline finished but the final dataframe is empty.")
            return pd.DataFrame()

        print(f"\n⏳ Formatting final schema. Total rows merged: {len(merged_df)}...")
        
        # Filter to target columns and sort
        existing_target_cols = [col for col in self.target_columns if col in merged_df.columns]
        merged_df = merged_df[existing_target_cols]
        merged_df = merged_df.sort_values(by=['player_name', 'date']).reset_index(drop=True)
        
        print(f"🎉 Pipeline complete! Final dataset shape: {merged_df.shape}")
        return merged_df