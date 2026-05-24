import pandas as pd
import json

class SessionFeatureExtractor:
    """
    A processing pipeline to extract, flatten, and aggregate subjective 
    training load data from session.json into daily timelines.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path

    def _load_and_flatten(self) -> pd.DataFrame:
        """Reads the JSON and flattens it from a dictionary into a tabular format."""
        print(f"Loading raw session data from {self.file_path}...")
        
        with open(self.file_path, 'r') as f:
            raw_data = json.load(f)
            
        records = []
        for player_name, sessions in raw_data.items():
            for session in sessions:
                records.append({
                    'player_name': player_name,
                    'date': session.get('date'),
                    'RPE': session.get('rpe', 0),
                    'duration obj': session.get('duration', 0),
                    'sRPE': session.get('srpe', 0)
                })
                
        return pd.DataFrame(records)

    def process_daily_features(self) -> pd.DataFrame:
        """
        Processes the flattened data, fixes date formats for merging, 
        and aggregates double-session days into a single daily load.
        """
        df = self._load_and_flatten()
        
        if df.empty:
            print("Warning: JSON file was empty.")
            return df
            
        # 1. Standardize Date Format for the Master Merge
        # The JSON has '17.03.2020'. We MUST convert this to '2020-03-17' 
        # so it perfectly matches your GPS and Wellness data when we join them.
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y').dt.strftime('%Y-%m-%d')
        
        # 2. Aggregate to the Daily Level
        # We group by date and player to combine multiple sessions in the same day
        print("Aggregating sessions to daily level...")
        daily_df = df.groupby(['date', 'player_name']).agg(
            daily_load=('sRPE', 'sum'),        # Total load for the day
            sRPE=('sRPE', 'sum'),              # Keep a column named sRPE as requested
            duration_obj=('duration obj', 'sum'), # Total duration of all sessions
            
            # For RPE, if they had 2 sessions (e.g., an RPE 4 and an RPE 8), 
            # we take the mathematical average to represent the daily feel.
            RPE=('RPE', 'mean')                
        ).reset_index()
        
        # Rename column back to exactly what you requested
        daily_df.rename(columns={'duration_obj': 'duration obj'}, inplace=True)
        
        # 3. Calculate Weighted RPE (Optional but recommended for Sports Science)
        # Instead of a pure mean, weighted RPE factors in the duration of each session.
        # daily_df['RPE_weighted'] = daily_df['daily_load'] / daily_df['duration obj']
        
        print("✅ Session data successfully processed into daily features.")
        return daily_df
