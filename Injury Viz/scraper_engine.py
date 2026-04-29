from seleniumbase import Driver
import pandas as pd
from bs4 import BeautifulSoup
import time
import random
import re
import io
import os

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class FBRefScraper:
    def __init__(self):
        self.teams = {
            "Arsenal": {"id": "18bb7c10", "slug": "Arsenal"},
            "Man_United": {"id": "19538871", "slug": "Manchester-United"},
            "Man_City": {"id": "b8fd03ef", "slug": "Manchester-City"},
            "Chelsea": {"id": "cff3d9bb", "slug": "Chelsea"},
            "Tottenham": {"id": "361ca564", "slug": "Tottenham-Hotspur"},
            "Liverpool": {"id": "822bd0ba", "slug": "Liverpool"}
        }
        
        self.target_columns = [
            "Date", "Time", "Comp", "Round", "Day", "Venue", "Result", 
            "GF", "GA", "Opponent", "Poss", "Attendance", "Captain", 
            "Formation", "Opp Formation"
        ]

        # data save location
        self.output_dir = "raw data"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _clean_data(self, df):
        """Applies Data Engineering best practices to clean the raw DataFrame."""
        cols_to_keep = [col for col in self.target_columns if col in df.columns]
        df = df[cols_to_keep].copy()
        
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        
        if 'date' in df.columns:
            df = df.dropna(subset=['date'])
            df = df[df['date'].astype(str).str.strip() != ''] 
            
        if 'attendance' in df.columns:
            df['attendance'] = df['attendance'].astype(str).str.replace(',', '', regex=False)
            df['attendance'] = pd.to_numeric(df['attendance'], errors='coerce')
            
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            
        return df

    def scrape_team_season(self, driver, team_name, season):
        """Fetches and parses the fixtures table using an undetectable browser."""
        team_info = self.teams.get(team_name)
        if not team_info:
            print(f"[!] Error: {team_name} is not configured.")
            return None
            
        url = f"https://fbref.com/en/squads/{team_info['id']}/{season}/all_comps/{team_info['slug']}-Stats-All-Competitions"
        print(f"[*] Fetching target: {team_name} | Season: {season}...")
        
        try:
            # Navigate to the URL using the undetected browser
            driver.get(url)
            
            # Pause to allow Cloudflare's JavaScript challenge to execute and pass
            time.sleep(random.uniform(2.5, 4.0))
            
            # Extract the fully rendered DOM
            html_content = driver.page_source
            
            # FBref hides data tables in HTML comments. Strip them out.
            html_content = re.sub(r'', '', html_content)
            soup = BeautifulSoup(html_content, 'lxml')
            
            table = soup.find("table", {"id": "matchlogs_for"})
            if not table:
                print(f"[!] Warning: 'matchlogs_for' table missing for {team_name} ({season}). Page may have been blocked.")
                return None
                
            df = pd.read_html(io.StringIO(str(table)))[0]
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
                
            df_cleaned = self._clean_data(df)
            return df_cleaned
            
        except Exception as e:
            print(f"[!] Data Processing Error on {team_name} {season}: {e}")
            return None

    def run_pipeline(self, start_year=2019, end_year=2025):
        """Orchestrates the scraping process iteratively."""
        seasons = [f"{year}-{year+1}" for year in range(start_year, end_year + 1)]
        
        # Initialize Undetected Chromedriver (UC) in headless mode
        print("[*] Booting Undetected Browser Engine...")
        driver = Driver(uc=True, headless=True)
        
        try:
            for team in self.teams.keys():
                for season in seasons:
                    df = self.scrape_team_season(driver, team, season)
                    
                    if df is not None and not df.empty:

                        filename = f"fixture_table_{team.lower()}_{season}.csv"
                        file_path = os.path.join(self.output_dir, filename)
                        
                        df.to_csv(file_path, index=False)
                        print(f"[+] SUCCESS: Saved {file_path} ({len(df)} match records)")
                    else:
                        print(f"[-] SKIPPED: No data extracted for {team} {season}")
                    
                    # Inter-request jitter
                    sleep_time = random.uniform(3.0, 5.5)
                    print(f"--> Sleeping {sleep_time:.2f}s to respect server load...\n")
                    time.sleep(sleep_time)
        finally:
            # Ensure the browser process is killed even if the script crashes
            print("[*] Shutting down browser engine.")
            driver.quit()

class TMAbsenceScraper:
    def __init__(self):
        self.output_dir = "raw data"
        self._prepare_directory()

        self.teams = {
            "Arsenal": {"id": "11", "slug": "arsenal-fc"},
            "Man_United": {"id": "985", "slug": "manchester-united"},
            "Man_City": {"id": "281", "slug": "manchester-city"},
            "Chelsea": {"id": "631", "slug": "chelsea-fc"},
            "Tottenham": {"id": "148", "slug": "tottenham-hotspur"},
            "Liverpool": {"id": "31", "slug": "fc-liverpool"}
        }

    def _prepare_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"[*] Created directory: {self.output_dir}")

    def _parse_status(self, td):
        """Maps Transfermarkt's backend CSS classes to the strict analytical legend."""
        classes = td.get('class', [])
        
        if 'ausfallzeiten_s' in classes:
            return 'Starting eleven'
        elif 'ausfallzeiten_e' in classes:
            return 'Substituted in'
        elif 'ausfallzeiten_k' in classes:
            return 'On the bench'
        elif 'ausfallzeiten_v' in classes or 'ausfallzeiten_a' in classes:
            return 'Absence/injury'
        else:
            # Covers 'ausfallzeiten_r' (Rest), 'bg_grey_afz' (U21 matches), and empty cells
            return 'Not included'

    def scrape_team_season(self, driver, team_name, start_year):
        team_info = self.teams.get(team_name)
        if not team_info:
            return None
            
        clearance_url = f"https://www.transfermarkt.co.za/{team_info['slug']}/startseite/verein/{team_info['id']}"
        data_url = f"https://www.transfermarkt.co.za/{team_info['slug']}/ausfallzeiten/verein/{team_info['id']}?reldata=GB1%26{start_year}"
        
        print(f"[*] Fetching Absence Data: {team_name} | Season: {start_year}...")
        print(f"    -> Target Link: {data_url}")
        
        try:
            # --- STEP 1: CLEARANCE ---
            driver.uc_open_with_reconnect(clearance_url, reconnect_time=4)
            time.sleep(random.uniform(3.0, 5.0))
            
            # --- STEP 2: COOKIE DISMISSAL ---
            try:
                iframe = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//iframe[starts-with(@id, 'sp_message_iframe')]"))
                )
                driver.switch_to.frame(iframe)
                button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Accept & continue')]"))
                )
                button.click()
                print("    -> [+] Auto-dismissed Cookie Banner.")
                time.sleep(2)
            except Exception:
                pass
            finally:
                driver.switch_to.default_content()
                
            # --- STEP 3: PIVOT ---
            driver.get(data_url)
            time.sleep(random.uniform(3.0, 5.0)) 
            
            # --- STEP 4: PARSE ---
            soup = BeautifulSoup(driver.page_source, 'lxml')
            
            # CRITICAL FIX: Target the specific ausfallzeiten-table
            table = soup.find("table", class_="ausfallzeiten-table")
            if not table:
                print(f"[!] Warning: Absence table missing. TM lacks data for {start_year}.")
                return None
                
            # Extract Headers Dynamically
            thead = table.find("thead")
            gw_headers = []
            for th in thead.find_all("th"):
                text = th.get_text(strip=True)
                if text.isdigit():
                    gw_headers.append(f"GW_{text}")
                    
            if not gw_headers:
                print(f"[-] No Gameweeks found for {team_name} {start_year}. Skipping.")
                return None

            # Extract Rows
            data = []
            tbody = table.find("tbody")
            rows = tbody.find_all("tr", recursive=False)
            
            for row in rows:
                player_tag = row.find("td", class_="hauptlink")
                if not player_tag:
                    continue 
                    
                player_name = player_tag.get_text(strip=True)
                
                # CRITICAL FIX: Filter out the duplicate 'hide' columns
                all_tds = row.find_all("td", recursive=False)
                visible_tds = [td for td in all_tds if 'hide' not in td.get('class', [])]
                
                # The first 3 visible tds are colour-bar, Name, and Position. Gameweeks start at index 3.
                gw_tds = visible_tds[3:]
                
                # Zip headers and cells
                row_data = {"Player": player_name}
                for gw_name, td in zip(gw_headers, gw_tds):
                    row_data[gw_name] = self._parse_status(td)
                    
                data.append(row_data)
                
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"[!] Data Processing Error on {team_name} {start_year}: {e}")
            return None

    def run_pipeline(self, start_year=2019, end_year=2025):
        seasons = [year for year in range(start_year, end_year + 1)]
       
        print("[*] Booting Undetected Browser Engine for Transfermarkt...")
        driver = Driver(uc=True, headless=True, window_size="1920,1080")
        
        try:
            for team in self.teams.keys():
                for year in seasons:
                    season_str = f"{year}-{year+1}"
                    filename = f"absence_table_{team.lower()}_{season_str}.csv"
                    file_path = os.path.join(self.output_dir, filename)
                    
                    # --- IDEMPOTENCY CHECK ---
                    # If the file is already in your folder, skip it to save IP reputation!
                    if os.path.exists(file_path):
                        print(f"[-] SKIPPED (Already Exists): {team} {season_str}")
                        continue
                        
                    df = self.scrape_team_season(driver, team, year)
                    
                    if df is not None and not df.empty:
                        df.to_csv(file_path, index=False)
                        print(f"[+] SUCCESS: Saved to {file_path} ({len(df)} players extracted)")
                    else:
                        print(f"[-] SKIPPED/FAILED: {team} {season_str}")
                    
                    # Increased jitter to evade rate-limiting bans
                    sleep_time = random.uniform(6.0, 9.5)
                    print(f"--> Cooling down for {sleep_time:.2f}s...\n")
                    time.sleep(sleep_time)
        finally:
            print("[*] Shutting down browser engine.")
            driver.quit()