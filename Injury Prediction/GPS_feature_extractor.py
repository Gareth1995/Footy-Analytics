import pandas as pd
import numpy as np
import glob
import os

class GPSFeatureExtractor:
    """
    A processing pipeline to extract aggregated physical features 
    from raw, high-frequency (50Hz) GPS tracking data.
    """
    
    def __init__(self, hz: int = 50, is_ms: bool = True, custom_zones: dict = None):
        self.hz = hz
        self.is_ms = is_ms
        
        # Pre-calculate time constants based on the sampling frequency
        self.time_per_row_sec = 1.0 / self.hz
        self.time_per_row_min = self.time_per_row_sec / 60.0
        self.time_per_row_hr  = self.time_per_row_sec / 3600.0
        
        # Define speed thresholds (Lower bound inclusive, Upper bound exclusive)
        # Allows user to pass custom zones if needed
        self.zones = custom_zones or {
            'lir': (0.0, 15.0),
            'mir': (15.0, 20.0),
            'hir': (20.0, 25.0),
            'spr': (25.0, 999.0) # 999 acts as an artificial ceiling
        }

    def process_single_file(self, file_path: str) -> dict:
        """Processes a single raw GPS CSV and returns a dictionary of features."""
        try:
            # Memory-safe loading: only grab what we need
            df = pd.read_parquet(file_path, columns=['player_name', 'speed'])
            
            if df.empty:
                return None
                
            player_name = df['player_name'].iloc[0]
            
            # Convert to km/h
            speed_col = df['speed'] * 3.6 if self.is_ms else df['speed']
            df['speed_kmh'] = speed_col
            
            # Calculate row-by-row distance
            df['dist_km'] = df['speed_kmh'] * self.time_per_row_hr
            
            total_time_min = len(df) * self.time_per_row_min # calculate total training duration in minutes
            total_dist_km = df['dist_km'].sum()
            
            # Map Zones dynamically based on initialized dictionary
            conditions = [
                (df['speed_kmh'] >= bounds[0]) & (df['speed_kmh'] < bounds[1])
                for bounds in self.zones.values()
            ]
            choices = list(self.zones.keys())
            df['zone'] = np.select(conditions, choices, default='lir')
            
            # Aggregate
            zone_stats = df.groupby('zone').agg(
                time_sec=('speed_kmh', 'count'), # calculates a count of all rows at the assigned speed
                distance_km=('dist_km', 'sum')   # calculates distance covered at the assigned speed
            )
            zone_stats['time_min'] = zone_stats['time_sec'] * self.time_per_row_min
            
            def get_val(z, metric):
                return zone_stats.loc[z, metric] if z in zone_stats.index else 0.0

            # Build final dictionary
            features = {
                'player_name': player_name,
                'Speed_km_h_mean': df['speed_kmh'].mean(),
                'Speed_km_h_max': df['speed_kmh'].max(),
                'Speed_km_h_std': df['speed_kmh'].std(),
                'Distance': total_dist_km,
                'distance_per_min': total_dist_km / total_time_min if total_time_min > 0 else 0,
                'total_time_min': total_time_min,
            }
            
            for z in self.zones.keys():
                t_min = get_val(z, 'time_min')
                features[f'sp_{z}_t'] = t_min
                features[f'sp_{z}_d'] = get_val(z, 'distance_km')
                features[f'sp_{z}_p'] = (t_min / total_time_min) if total_time_min > 0 else 0
                
            return features
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    # def process_directory(self, dir_path: str) -> pd.DataFrame:
    #     """Loops through a directory, processing all CSVs into a master DataFrame."""
    #     all_files = glob.glob(os.path.join(dir_path, "*.parquet"))
    #     print(f"Found {len(all_files)} files. Initializing batch extraction...")
        
    #     results = []
    #     for i, file in enumerate(all_files):
    #         session_data = self.process_single_file(file)
    #         if session_data:
    #             # Optional: Extract date from filename if your filenames have dates!
    #             # session_data['date'] = os.path.basename(file).split('_')[0] 
    #             results.append(session_data)
                
    #         if i > 0 and i % 50 == 0:
    #             print(f"  -> Processed {i} files...")
                
    #     print("✅ Batch processing complete.")
    #     return pd.DataFrame(results)

    def process_directory(self, base_dir: str) -> pd.DataFrame:
        """Loops through a nested directory recursively, processing all Parquet files."""
        
        # The '**' combined with recursive=True tells Python to tunnel through 
        # the 2020-06 folder, into the 2020-06-01 folder, and grab the files.
        search_path = os.path.join(base_dir, "**", "*.parquet")
        all_files = glob.glob(search_path, recursive=True)
        
        print(f"Found {len(all_files)} Parquet files across all months. Starting batch extraction...")
        
        results = []
        for i, file in enumerate(all_files):
            session_data = self.process_single_file(file)
            
            if session_data:
                # Extract the date from the filename! 
                # os.path.basename gets just the filename (e.g., '2020-06-01-TeamB...')
                # [:10] slices exactly the first 10 characters (YYYY-MM-DD)
                filename = os.path.basename(file)
                session_data['date'] = filename[:10]
                
                results.append(session_data)
                
            if i > 0 and i % 50 == 0:
                print(f"  -> Processed {i} / {len(all_files)} files...")
                
        print("✅ Batch processing complete.")
        return pd.DataFrame(results)
