import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler

class DeepHitPreprocessor:
    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.standard_scaler = StandardScaler()
        self.power_scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        self.clip_bounds = {}

        self.standard_cols = [
            'Fatigue', 'Mood', 'Readiness', 'Sleep duration', 'Soreness', 'Stress', 'RPE',
            'Speed_km_h_mean', 'Speed_km_h_max', 'Speed_km_h_std', 
            'sp_lir_p', 'sp_mir_p', 'sp_hir_p', 'sp_spr_p',
            'Distance', 'distance_per_min', 'total_time_min', 
            'sp_lir_t', 'sp_lir_d', 'sp_mir_t', 'sp_mir_d', 
            'sp_hir_t', 'sp_hir_d', 'sp_spr_t', 'sp_spr_d'
        ]
        
        self.skewed_cols = [
            'ATL', 'Weekly load', 'Monotony', 'Strain', 'ACWR', 'CTL28', 'CTL42', 
            'daily_load', 'sRPE', 'duration obj'
        ]

    def _filter_existing_cols(self, df: pd.DataFrame):
        self.process_std = [c for c in self.standard_cols if c in df.columns]
        self.process_skew = [c for c in self.skewed_cols if c in df.columns]

    def _safe_fit_transform(self, df: pd.DataFrame, columns: list, scaler) -> pd.DataFrame:
        """Fits and transforms only on non-zero, non-NaN rows to preserve variance."""
        for col in columns:
            # Create a mask for rows that actually have training data
            valid_mask = (df[col].notna()) & (df[col] != 0)
            
            if valid_mask.sum() > 0:
                # Extract the valid data, reshape for sklearn, and fit_transform
                valid_data = df.loc[valid_mask, col].values.reshape(-1, 1)
                transformed_data = scaler.fit_transform(valid_data)
                
                # Place the transformed data back into the dataframe
                df.loc[valid_mask, col] = transformed_data.flatten()
        return df

    def _safe_transform(self, df: pd.DataFrame, columns: list, scaler) -> pd.DataFrame:
        """Transforms only on non-zero, non-NaN rows using pre-fit scaler."""
        for col in columns:
            valid_mask = (df[col].notna()) & (df[col] != 0)
            
            if valid_mask.sum() > 0:
                valid_data = df.loc[valid_mask, col].values.reshape(-1, 1)
                transformed_data = scaler.transform(valid_data)
                df.loc[valid_mask, col] = transformed_data.flatten()
        return df

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        print("🔧 Fitting preprocessor safely on Training Data...")
        df = X_train.copy()
        self._filter_existing_cols(df)
        
        # 1. Calculate clipping bounds (Ignore 0s and NaNs)
        for col in self.process_std + self.process_skew:
            valid_data = df.loc[(df[col].notna()) & (df[col] != 0), col]
            if len(valid_data) > 0:
                lower_bound = valid_data.quantile(self.lower_q)
                upper_bound = valid_data.quantile(self.upper_q)
                if col in self.process_skew:
                    lower_bound = valid_data.min()
                self.clip_bounds[col] = (lower_bound, upper_bound)
                
                # Apply clipping to valid data
                df.loc[valid_data.index, col] = valid_data.clip(lower=lower_bound, upper=upper_bound)

        # 2. Safe Fit & Transform (Only acts on non-zero days)
        if self.process_skew:
            df = self._safe_fit_transform(df, self.process_skew, self.power_scaler)
        if self.process_std:
            df = self._safe_fit_transform(df, self.process_std, self.standard_scaler)
            
        # 3. Final Imputation: Ensure all NaNs and skipped 0s are explicitly 0.0 for DeepHit
        df[self.process_std + self.process_skew] = df[self.process_std + self.process_skew].fillna(0.0)
        
        return df

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        print("🚀 Applying safe transformations to Test Data...")
        df = X_test.copy()
        self._filter_existing_cols(df)
        
        # 1. Apply memorized clipping bounds to valid data
        for col in self.process_std + self.process_skew:
            if col in self.clip_bounds:
                valid_mask = (df[col].notna()) & (df[col] != 0)
                if valid_mask.sum() > 0:
                    lower_bound, upper_bound = self.clip_bounds[col]
                    df.loc[valid_mask, col] = df.loc[valid_mask, col].clip(lower=lower_bound, upper=upper_bound)

        # 2. Safe Transform (Only acts on non-zero days)
        if self.process_skew:
            df = self._safe_transform(df, self.process_skew, self.power_scaler)
        if self.process_std:
            df = self._safe_transform(df, self.process_std, self.standard_scaler)
            
        # 3. Final Imputation
        df[self.process_std + self.process_skew] = df[self.process_std + self.process_skew].fillna(0.0)
        
        return df
    