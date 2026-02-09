# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 13:34:05 2025

@author: Lorenzo CTR Flores-Y
"""
#%%
# 0.INSTALL LIBRARIES & UPLOAD CWP DATA
# Installing  cx_Oracle (intalled on anaconda_prompt)
import cx_Oracle

#%%
import os 
# Tell Python where the Oracle Instant Client is
os.add_dll_directory(r"C:\oracle_spyder\instantclient_23_9")

#%%
# Defina the path for the path for the oracle drivers
import os
os.environ['PATH'] = r"C:\oracle_spyder\instantclient_23_9;" + os.environ['PATH']

#%%
# Add Oracle Instant Client folder first
os.add_dll_directory(r"C:\oracle_spyder\instantclient_23_9")

# os.environ['PATH'] = r"C:\oracle_spyder\instantclient_23_9;" + os.environ['PATH']  # optional

print("cx_Oracle version:", cx_Oracle.version)

# Connection Credentials 
dsn = cx_Oracle.makedsn("pdbsamc505.faa.gov", 1521, service_name="ndc") 
conn = cx_Oracle.connect(user="LFYANEZ", password="SummerVerano2027%", dsn=dsn) #enter your user name and password to connect to the Oracle database

print("Connected successfully!")

#%%
# Create a cursor
cur = conn.cursor()

# Fetch just 10 rows to test the connection
query = """
SELECT JCN, JCN_DESC, JCN_STATUS, CIP, LOCID_CD, PROJ_CD, ORG_SVC_AREA,
       JCN_STAT_DT, COMPLTN_DT, STATE, PA_EXECUTION_DT, MOST_DETAILED_COST_EST,
       PA_PLANNING_FY, TIER, FACTYP_CD
FROM CWP_PROJ_COLL_EXT_VW
WHERE ROWNUM <= 10
"""
cur.execute(query)

# Fetch the rows
rows = cur.fetchall()

# Print them to confirm
for row in rows:
    print(row)

#%%
# --- 1.CWP TABALE DEVELOPMENT ---
# Import to a pandas dataframe 
import pandas as pd

query = """
SELECT JCN, JCN_DESC, JCN_STATUS, CIP, LOCID_CD, PROJ_CD, ORG_SVC_AREA,
       JCN_STAT_DT, COMPLTN_DT, STATE, PA_EXECUTION_DT, MOST_DETAILED_COST_EST,
       PA_PLANNING_FY, TIER, FACTYP_CD
FROM CWP_PROJ_COLL_EXT_VW
"""

df = pd.read_sql(query, con=conn)
print(df.dtypes)
#%%
# Convert all column names to snake_case
df.columns = df.columns.str.strip() \
                       .str.lower() \
                       .str.replace(' ', '_') \
                       .str.replace('-', '_')
# Show the results 
df.info()
#%%
# Filter FIP_CIPs
# Create the list of cips 
cip_list = [
    "F01.01-00", "F01.02-00", "F06.01-00", "F11.01-02", "F12.00-00",
    "F13.01-00", "F13.02-00", "F13.03-00", "F13.04-02", "F20.01-01",
    "F24.01-02", "F24.02-01", "F26.01-01", "F31.01-01", "F34.01-01",
    "S04.02-03", "F35.02-01", "F35.04-01", "F35.05-01", "F35.06-01",
    "F35.07-01", "F35.08-01", "F35.09-01", "F35.11-01", "F35.12-01",
    "F35.13-01", "F35.14-01", "F35.15-01", "F35.16-01", "F35.17-01",
    "F35.18-01"
]

# Keep only the FIP_CIPs
df_cwp = df[df["cip"].isin(cip_list)]

# Show the results
df_cwp.info()

print(df_cwp.shape)
#%%
# Remove jcn_status N, U, W, X
df_cwp = df_cwp[~df_cwp["jcn_status"].isin(["N", "U", "W", "X"])]

# Show all unique values to confirm
print(df_cwp["jcn_status"].unique())

#%%
# Count null values in compltn_dt
null_count = df_cwp["compltn_dt"].isnull().sum()
print("Number of null compltn_dt values:", null_count)

#%% 
# Convert compltn_dt to datetime (empty strings become NaT)
df_cwp["compltn_dt"] = pd.to_datetime(df_cwp["compltn_dt"], errors="coerce")

# Create date_flag with 3 categories
def date_category(x):
    if pd.isna(x):
        return "EMPTY"
    elif x > pd.Timestamp("2016-09-30"):
        return "IN"
    else:
        return "OUT"

df_cwp["date_flag"] = df_cwp["compltn_dt"].apply(date_category)

# Count the occurrences of each category
date_flag_counts = df_cwp["date_flag"].value_counts()
print(date_flag_counts)

#%%
# Keep only rows where date_flag is not OUT
df_cwp_filtered = df_cwp[df_cwp["date_flag"] != "OUT"]

# Check the counts of remaining categories
print(df_cwp_filtered["date_flag"].value_counts())

#%%
# Define the completed statuses
completed_statuses = ["C", "D", "F", "MC", "MD"]

# Create a new column empty_flag
df_cwp_filtered["empty_flag"] = df_cwp_filtered.apply(
    lambda row: "YES" if row["date_flag"] == "EMPTY" and row["jcn_status"] in completed_statuses else "NO",
    axis=1
)

# Ccheck the counts
print(df_cwp_filtered["empty_flag"].value_counts())
#%%
#  Exclude rows where empty_flag is "YES"
df_cwp_final = df_cwp_filtered[df_cwp_filtered["empty_flag"] != "YES"]

#  Drop the helper columns date_flag and empty_flag
df_cwp_final = df_cwp_final.drop(columns=["date_flag", "empty_flag"])

# Check the final DataFrame
print("Final number of rows:", len(df_cwp_final))

print(df_cwp_final['jcn_status'].value_counts())
#%%
# --- 2.LOAD FIP_FY17-FY27 FILE ---
import os

# Check current working directory
print("Current working directory:", os.getcwd())
#%%
# Change working directory (if needed)
os.chdir(r"C:\Users\Lorenzo CTR Flores-Y\Documents\python_flow")
print("New working directory:", os.getcwd())

#%%
# Read the CSV file (replace 'your_file.csv' with your actual file name)
df_fip_fy = pd.read_csv('FIP_FY17-FY27_AUG2025.csv')

# Display the first few rows to confirm it loaded correctly
df_fip_fy.info()
#%%
# Convert all column names to snake_case
df_fip_fy.columns = df_fip_fy.columns.str.strip() \
                       .str.lower() \
                       .str.replace(' ', '_') \
                       .str.replace('-', '_')
# Show the results 
df_fip_fy.info()
print(len(df_fip_fy))

#%%
# Create a duplicate of df_fip_fy
df_fy_v2 = df_fip_fy.copy()

df_fy_v2.info()

#%% Clean the file
# List of columns to remove
cols_to_drop = [
    "fip_baseline", "jcn_desc", "jcn_status", "cip", "locid_cd",
    "org_svc_area", "proj_cd", "jcn_stat_dt", "compltn_dt", "state",
    "pa_execution_dt", "most_detailed_cost_est", "pa_planning_fy",
    "tier", "multi-year"
]

# Drop the columns
df_fip_fy = df_fip_fy.drop(columns=cols_to_drop, errors="ignore")

# Verify result
df_fip_fy.info()

#%%
# --- 3.JOIN FIP_CWP & FIP_FY17-FY27 FOR FIP_ADDITIONS ---

# Perform a left join first
merged = pd.merge(
    df_cwp_final,
    df_fip_fy,
    on="jcn",
    how="left",
    indicator=True
)

# Keep only rows from df_cwp_final that did NOT match in df_fip_fy
df_cwp_add = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

# Show the results 
df_cwp_add.info()

# Check the results
print("Rows in df_cwp_final:", len(df_cwp_final))
print("Rows in df_fip_fy:", len(df_fip_fy))
print("Rows in df_cwp_add (unmatched):", len(df_cwp_add))

print(df_cwp_add['jcn_status'].value_counts())

df_cwp_add.info()

#%%
# --- 4. JOIN FIP_ADDITIONS WITH SHORTFALLS ---

# Read the CSV file (replace 'your_file.csv' with your actual file name)
df_shortfalls = pd.read_csv('fip_shortfalls.csv')

# Display the first few rows to confirm it loaded correctly
print(df_shortfalls.head())

df_shortfalls.info()
#%%
# Convert all column names to snake_case
df_shortfalls.columns = df_shortfalls.columns.str.strip() \
                       .str.lower() \
                       .str.replace(' ', '_') \
                       .str.replace('-', '_')
# Show the results 
df_shortfalls.info()
#%%
# Remove all the parametes and only keep JCN
df_shortfalls = df_shortfalls[['jcn']]

# Add new column with constant value
df_shortfalls['fip_add_shortfall'] = "YES"

print(df_shortfalls.head())
print("Rows in df_shortfalls:", len(df_shortfalls))
#%%
# Perform a left join addition & shortfalls 
df_cwp_add_shortfalls = pd.merge(
    df_cwp_add,
    df_shortfalls,
    on="jcn",
    how="left"
)

# Verify row counts
print("Rows in df_cwp_add:", len(df_cwp_add))
print("Rows in df_shortfalls:", len(df_shortfalls))
print("Rows in df_cwp_add_shortfalls (left join result):", len(df_cwp_add_shortfalls))

print(df_cwp_add_shortfalls['jcn_status'].value_counts())

df_cwp_add_shortfalls.info()
#%%
# Add fip_baseline column
df_fip_fy["fip_baseline"] = "YES"
df_cwp_add_shortfalls["fip_baseline"] = "NO"

# Force multi_year = "NO" for all cwp_add_shortfalls
df_cwp_add_shortfalls["multi_year"] = "NO"
#%%
df_cwp_add_shortfalls.info()
#%%
# Keep only the neded parameters
df_cwp_subset = df[["jcn", "jcn_status", "most_detailed_cost_est", "jcn_stat_dt" ]]

df_cwp_subset.info()

print(len(df_cwp_subset))
print(df_cwp_subset['jcn_status'].value_counts())

df_cwp_subset.info()
#%%
#%% NEW UPDATE
# Keep only the columns we need from df_fy_v2
df_fy_v2_subset = df_fy_v2[['jcn', 'pa_execution_dt', 'pa_planning_fy', 'fip_baseline', 'fip_fy', 'fip_cip', 'baseline_mdce']]

df_fy_v2_subset.info()

#%% df_fip_cwp_subset_updated
# Left join with df_fip_fy as the left table
df_fip_cwp_subset_updated = df_fy_v2_subset.merge(
    df_cwp_subset,
    on="jcn",
    how="left",
    suffixes=("", "_cwp")
)

# Validate with row counts
print("Row count df_fip_fy:", len(df_fip_fy))
print("Row count df_cwp_subset:", len(df_cwp_subset))
print("Row count df_fip_cwp_subset_updated:", len(df_fip_cwp_subset_updated))

#%% UPPDATED df_fip_cwp_subset
# Left join with df_fip_fy as the left table
df_fip_cwp_subset_updated = df_fy_v2_subset.merge(
    df_cwp_subset,
    on="jcn",
    how="left",
    suffixes=("", "_cwp")
)

# Validate with row counts
print("Row count df_fip_fy_v2_subset:", len(df_fy_v2_subset))
print("Row count df_cwp_subset:", len(df_cwp_subset))
print("Row count df_fip_cwp_subset_updated:", len(df_fip_cwp_subset_updated))


#%%
#%% ORIGINAL df_fip_cwp_subset
# Left join with df_fip_fy as the left table
df_fip_cwp_subset = df_fip_fy.merge(
    df_cwp_subset,
    on="jcn",
    how="left",
    suffixes=("", "_cwp")
)

# Validate with row counts
print("Row count df_fip_fy:", len(df_fip_fy))
print("Row count df_cwp_subset:", len(df_cwp_subset))
print("Row count df_fip_cwp_subset:", len(df_fip_cwp_subset))

print(df_fip_cwp_subset['jcn_status'].value_counts())

#%%
# --- 5.UNION FIP_FY & CWP_ADD ---

# Union (row-wise append)
df_union_fip_add = pd.concat([df_cwp_add_shortfalls, df_fip_cwp_subset_updated], ignore_index=True)

# Check the result
print(len(df_cwp_add_shortfalls))
print(len(df_fip_fy))              # should be 9510
print(len(df_union_fip_add))               

print(df_union_fip_add['jcn_status'].value_counts(dropna=False))
#%%
# Validate the union 
df_union_fip_add.info()

#%%
# Count unique jcn values
unique_jcn_count = df_union_fip_add["jcn"].nunique()
print("Unique jcn count:", unique_jcn_count)

# Count total jcn rows
total_jcn_rows = df_union_fip_add["jcn"].count()
print("Total jcn rows:", total_jcn_rows)

# Distribution of multi_year
print("\nmulti_year counts:")
print(df_union_fip_add["multi_year"].value_counts(dropna=False))

# Distribution of fip_baseline
print("\nfip_baseline counts:")
print(df_union_fip_add["fip_baseline"].value_counts(dropna=False))
#%%
# Shortfalls 
# Replace any NaN with NO (safety check)
df_union_fip_add["fip_add_shortfall"] = df_union_fip_add["fip_add_shortfall"].fillna("NO")
print(df_union_fip_add["fip_add_shortfall"].value_counts(dropna=False))
#%% 
# --- 6.GET OPPM CHANGE REASONS & MULTI-YEAR --- 

# Read the CSV file (replace 'your_file.csv' with your actual file name)
df_my = pd.read_csv('my_fy17_fy28_v1.csv')

df_my = df_my[["JCN", "MULTI_YEAR", "FIP_FY"]]
df_my.info()

#%%
# Convert all column names to snake_case
df_my.columns = df_my.columns.str.lower()

# Show the results 
df_my.info()

#%%
# Count and Exclude the NO values on Multi-Year 
df_my['multi_year'].value_counts()

df_my= df_my[df_my['multi_year'] != "NO"]

print(df_my['multi_year'].value_counts())
#%% 
# Sort by jcn and fip_fy descending
df_my = df_my.sort_values(['jcn', 'fip_fy'], ascending=[True, False])

# Keep only the latest fip_fy per jcn
df_my = df_my.groupby('jcn', group_keys=False).head(1)

# Validate
print("Number of rows:", len(df_my))
print(df_my['jcn'].value_counts())  # all values should be 1

df_my.info()
#%%
# Drop fip_fy
df_my = df_my.drop(columns=['fip_fy'])

df_my.info()

#%%
# Get the Change Reasons dataset and format it 
df_change_reasons = pd.read_csv('fip_change_tracking_sep.csv')

# Convert all column names to snake_case
df_change_reasons.columns = df_change_reasons.columns.str.lower()

# Select only the parameters needed for the analysis 
df_change_reasons = df_change_reasons[["jcn", "items", "fip_year", "fip_changenotes", "fip_baselinechangereasons"]]

df_change_reasons.info()

#%%
# Exclude all the null values
df_change_reasons = df_change_reasons[df_change_reasons['jcn'].notna()]

print("Number of rows:", len(df_change_reasons))


# Rename items to fip_jcn_description 
df_change_reasons = df_change_reasons.rename(columns={'items': 'fip_jcn_description'})

df_change_reasons.info()

#%%
# Delete duplicates 
df_change_reasons = df_change_reasons.drop_duplicates(subset='jcn', keep='first')

print("Number of rows:", len(df_change_reasons))

#%%
df_my_change_reasons = pd.merge(
    df_my,
    df_change_reasons,
    on='jcn',
    how='outer'
)

# Optional: check the result
print("Number of rows after full outer join:", len(df_my_change_reasons))

df_my_change_reasons.info()

print(df_my_change_reasons['multi_year'].value_counts())

df_my_change_reasons.info()
#%%
# 7. --- JOIN OPPM CHANGE REASONS - MULTI-YEAR WITH FIP --- 

#Validating the correct table
print(df_union_fip_add['fip_baseline'].value_counts())

# Validting the other table 
print(len(df_union_fip_add))


# Validting the other table 
print(len(df_my_change_reasons))

#%%
df_fip_cwp_oppm = df_union_fip_add.merge(
    df_my_change_reasons,
    on='jcn',
    how='left'
)

df_fip_cwp_oppm.info()
#%%
# Drop multi_year_x 
df_fip_cwp_oppm = df_fip_cwp_oppm.drop(columns='multi_year_x')

# Rename mulit year and fill all the non values for NO
df_fip_cwp_oppm['multi_year'] = df_fip_cwp_oppm['multi_year_y'].fillna('NO')

df_fip_cwp_oppm = df_fip_cwp_oppm.drop(columns='multi_year_y')

# Validate the changes 
df_fip_cwp_oppm.info()

print(df_fip_cwp_oppm['multi_year'].value_counts())

print(df_fip_cwp_oppm['fip_baseline'].value_counts())


#%%
# Validate the table 
print(df_fip_cwp_oppm['jcn_status'].value_counts(dropna=False))

print(df_fip_cwp_oppm['fip_baseline'].value_counts(dropna=False))


print(
    df_fip_cwp_oppm.pivot_table(
        index='jcn_status',
        columns='fip_baseline',
        aggfunc='size',
        fill_value=0
    )
    .assign(_Total=lambda x: x.sum(axis=1))
    .sort_values('_Total', ascending=False)
    .drop(columns='_Total')
)

#%%
# 8. ---SBIT STATUS --- 
# Define mapping dictionary
status_mapping = {
    "C": "COMPLETED",
    "D": "COMPLETED",
    "F": "COMPLETED",
    "MC": "COMPLETED",
    "MD": "COMPLETED",
    "M": "IN PROGRESS",
    "P": "IN PROGRESS",
    "R": "IN PROGRESS",
    "S": "IN PROGRESS",
    "A": "IN PROGRESS",
    "H": "UNPLANNED",
    "Z": "UNPLANNED",
    "W": "CANCELLED",
    "X": "CANCELLED",
    "U": "CANCELLED"
}

# Apply mapping to create a new column
df_fip_cwp_oppm["sbit_status"] = df_fip_cwp_oppm["jcn_status"].map(status_mapping)


# Make the NaN values Canceled 
df_fip_cwp_oppm["sbit_status"] = (
    df_fip_cwp_oppm["jcn_status"]
    .map(status_mapping)
    .fillna("CANCELLED")
)

# Validate the SBIT_STATUS 
print(df_fip_cwp_oppm['sbit_status'].value_counts(dropna=False))

#%%
# 9. ---CIP CALCULATED  ----
df_fip_cwp_oppm['cip'] = df_fip_cwp_oppm['cip'].combine_first(df_fip_cwp_oppm['fip_cip'])

print(df_fip_cwp_oppm['cip'].value_counts(dropna=False))

df_fip_cwp_oppm.info()
#%%
#10. --- RANK---
df_fip_cwp_oppm['row_number'] = (
    df_fip_cwp_oppm
    .sort_values(['jcn', 'fip_year'], ascending=[True, False])  # sort by jcn, then fip_year descending
    .groupby('jcn')
    .cumcount() + 1  # row_number starting at 1
)

print(df_fip_cwp_oppm['row_number'].value_counts(dropna=False))
#%% 
iija_values = [
    'F35.02-01', 'F35.04-01', 'F35.05-01', 'F35.06-01',
    'F35.07-01', 'F35.08-01', 'F35.09-01', 'F35.11-01',
    'F35.12-01', 'F35.13-01', 'F35.14-01', 'F35.15-01',
    'F35.16-01', 'F35.17-01', 'F35.18-01'
]

df_fip_cwp_oppm['cip_category'] = df_fip_cwp_oppm['cip'].apply(
    lambda x: 'IIJA' if x in iija_values else 'LEGACY'
)
print(df_fip_cwp_oppm['cip_category'].value_counts(dropna=False))

#%%
#11. --- SBIT NOTES --- 
# FIP Column 
# List of cip values considered as 'in'
cip_list = [
    'f01.01-00', 'f01.02-00', 'f06.01-00', 'f11.01-02', 'f12.00-00',
    'f13.01-00', 'f13.02-00', 'f13.03-00', 'f13.04-02', 'f20.01-01',
    'f24.01-02', 'f24.02-01', 'f26.01-01', 'f31.01-01', 'f34.01-01',
    's04.02-03'
]

# Create new column 'fip' based on parameter 'cip'
df_fip_cwp_oppm['fip'] = df_fip_cwp_oppm['cip'].apply(
    lambda x: 'in' if str(x).lower() in cip_list else 'out'
)
print(df_fip_cwp_oppm['fip'].value_counts(dropna=False))

# Create new colunn 'bils' based on parameter 'cip'
# list of cip values considered as 'in' for bils
bils_list = [
    'f35.02-01', 'f35.04-01', 'f35.05-01', 'f35.06-01', 'f35.07-01',
    'f35.08-01', 'f35.09-01', 'f35.11-01', 'f35.12-01', 'f35.13-01',
    'f35.14-01', 'f35.15-01', 'f35.16-01', 'f35.17-01', 'f35.18-01'
]

# Create new column 'bils' based on parameter 'cip'
df_fip_cwp_oppm['bils'] = df_fip_cwp_oppm['cip'].apply(
    lambda x: 'in' if str(x).lower() in bils_list else 'out'
)
print(df_fip_cwp_oppm['bils'].value_counts(dropna=False))

# Create new column x&w
# values considered as 'in' for x&w
xw_list = ['x', 'w', 'u']

# create new column 'x&w' based on column 'jcn_status-1'
# values considered as 'in' for x&w
xw_list = ['x', 'w', 'u']

# create new column 'x&w' including NaN as 'in'
df_fip_cwp_oppm['x&w'] = df_fip_cwp_oppm['jcn_status'].apply(
    lambda x: 'in' if (pd.isna(x) or str(x).lower() in xw_list) else 'out'
)

print(df_fip_cwp_oppm['x&w'].value_counts(dropna=False))
#%%
import numpy as np
import pandas as pd

# ensure pa_execution_dt is datetime
df_fip_cwp_oppm['pa_execution_dt'] = pd.to_datetime(df_fip_cwp_oppm['pa_execution_dt'], errors='coerce')

#%%
# define conditions
conditions = [
    # CANCELLED
    (df_fip_cwp_oppm['x&w'] == 'in'),
    
    # MOVED TO IIJA
    (df_fip_cwp_oppm['bils'] == 'in') & (df_fip_cwp_oppm['x&w'] == 'out'),
    
    # MOVED TO OTHER CIP
    (df_fip_cwp_oppm['fip'] == 'out') & (df_fip_cwp_oppm['bils'] == 'out') & (df_fip_cwp_oppm['x&w'] == 'out'),
    
    # DELAYED by year rules
    (df_fip_cwp_oppm['x&w'] == 'out') & (df_fip_cwp_oppm['pa_planning_fy'] == '2017') & (df_fip_cwp_oppm['pa_execution_dt'] > '2018-03-31'),
    (df_fip_cwp_oppm['x&w'] == 'out') & (df_fip_cwp_oppm['pa_planning_fy'] == '2018') & (df_fip_cwp_oppm['pa_execution_dt'] > '2019-03-31'),
    (df_fip_cwp_oppm['x&w'] == 'out') & (df_fip_cwp_oppm['pa_planning_fy'] == '2019') & (df_fip_cwp_oppm['pa_execution_dt'] > '2020-03-31'),
    (df_fip_cwp_oppm['x&w'] == 'out') & (df_fip_cwp_oppm['pa_planning_fy'] == '2020') & (df_fip_cwp_oppm['pa_execution_dt'] > '2021-03-31'),
    (df_fip_cwp_oppm['x&w'] == 'out') & (df_fip_cwp_oppm['pa_planning_fy'] == '2021') & (df_fip_cwp_oppm['pa_execution_dt'] > '2022-03-31'),
    (df_fip_cwp_oppm['x&w'] == 'out') & (df_fip_cwp_oppm['pa_planning_fy'] == '2022') & (df_fip_cwp_oppm['pa_execution_dt'] > '2023-03-31'),
    (df_fip_cwp_oppm['x&w'] == 'out') & (df_fip_cwp_oppm['pa_planning_fy'] == '2023') & (df_fip_cwp_oppm['pa_execution_dt'] > '2024-03-31'),
    (df_fip_cwp_oppm['x&w'] == 'out') & (df_fip_cwp_oppm['pa_planning_fy'] == '2024') & (df_fip_cwp_oppm['pa_execution_dt'] > '2025-03-31'),
    (df_fip_cwp_oppm['x&w'] == 'out') & (df_fip_cwp_oppm['pa_planning_fy'] == '2025') & (df_fip_cwp_oppm['pa_execution_dt'] > '2026-03-31'),
    (df_fip_cwp_oppm['x&w'] == 'out') & (df_fip_cwp_oppm['pa_planning_fy'] == '2026') & (df_fip_cwp_oppm['pa_execution_dt'] > '2027-03-31'),
    
    # NULL DELAYED
    (df_fip_cwp_oppm['pa_execution_dt'].isna())
]

# choices must match number of conditions
choices = [
    "CANCELLED",              # 1
    "MOVED TO IIJA",          # 2
    "MOVED TO OTHER CIP",     # 3
    "DELAYED", "DELAYED", "DELAYED", "DELAYED",  # 2017–2020
    "DELAYED", "DELAYED", "DELAYED", "DELAYED",  # 2021–2024
    "DELAYED", "DELAYED",     # 2025–2026
    "DELAYED"                 # NULL
]


# apply with np.select
df_fip_cwp_oppm['sbit_notes'] = np.select(conditions, choices, default="NO CHANGE")

print(df_fip_cwp_oppm['sbit_notes'].value_counts(dropna=False))
print(df_fip_cwp_oppm['sbit_status'].value_counts(dropna=False))

#%%
#12. --- FIP_ADDITION_CAT ---
import numpy as np
import pandas as pd

# Ensure text columns are strings
df_fip_cwp_oppm['jcn_desc'] = df_fip_cwp_oppm['jcn_desc'].astype(str)
df_fip_cwp_oppm['proj_cd'] = df_fip_cwp_oppm['proj_cd'].astype(str)

# Define conditions
conditions = [
    # CAB_GLASS
    df_fip_cwp_oppm['jcn_desc'].str.contains('CAB GLASS', case=False, na=False) |
    df_fip_cwp_oppm['jcn_desc'].str.contains('WINDOW PANELS', case=False, na=False) |
    df_fip_cwp_oppm['jcn_desc'].str.contains('WINDOW GLASS', case=False, na=False) |
    df_fip_cwp_oppm['jcn_desc'].str.contains('WINDOW PANEL', case=False, na=False) |
    df_fip_cwp_oppm['jcn_desc'].str.contains('CAB PANEL', case=False, na=False) |
    df_fip_cwp_oppm['jcn_desc'].str.contains('CAB PANELS', case=False, na=False),
    
    # CAB_SHADE
    df_fip_cwp_oppm['jcn_desc'].str.contains('CAB SHADE', case=False, na=False) |
    df_fip_cwp_oppm['jcn_desc'].str.contains('CAB SHADES', case=False, na=False) |
    df_fip_cwp_oppm['jcn_desc'].str.contains('WINDOW SHADE', case=False, na=False) |
    df_fip_cwp_oppm['jcn_desc'].str.contains('WINDOW SHADES', case=False, na=False),
    
    # ERMS
    df_fip_cwp_oppm['proj_cd'].isin(['98310684', '9831J684']),
    
    # BATTERIES
    df_fip_cwp_oppm['proj_cd'].isin(['98310686', '98310688', '98310695', '9831J695', '9831J696'])
]

# Corresponding choices
choices = [
    'CAB_GLASS',
    'CAB_SHADE',
    'ERMS',
    'BATTERIES'
]

# Apply conditions to create new column
df_fip_cwp_oppm['fip_additions_cat'] = np.select(conditions, choices, default='NO_CAT')

# Validate 
print(df_fip_cwp_oppm['fip_additions_cat'].value_counts(dropna=False))
#%%
# 13. --- DUPlICATION COUNTS --- 
# Count duplicates of each JCN
df_fip_cwp_oppm['dup_counts'] = df_fip_cwp_oppm.groupby('jcn')['jcn'].transform('count')

# Validate 
print(df_fip_cwp_oppm['dup_counts'].value_counts(dropna=False))

df_fip_cwp_oppm.info()
#%%
#14.---FIP_INSTANCE ---
# Create jcn_instances based on dup_counts
df_fip_cwp_oppm['jcn_instances'] = df_fip_cwp_oppm['dup_counts'].apply(
    lambda x: 'DUPLICATE' if x > 1 else 'UNIQUE'
)

# Validate 
print(df_fip_cwp_oppm['jcn_instances'].value_counts(dropna=False))

#%%
df_fip_cwp_oppm.info()
print(df_fip_cwp_oppm['fip_fy'].value_counts(dropna=False))

print(
    df_fip_cwp_oppm[df_fip_cwp_oppm['fip_baseline'] == 'YES']['fip_year']
    .value_counts(dropna=False)
)
#%%
# Convert fip_fy to numeric (ignores errors, keeps NaN), then to Int where possible, then to string
df_fip_cwp_oppm['fip_fy'] = pd.to_numeric(df_fip_cwp_oppm['fip_fy'], errors='coerce')
df_fip_cwp_oppm['fip_fy'] = df_fip_cwp_oppm['fip_fy'].apply(lambda x: str(int(x)) if pd.notna(x) else 'NaN')

# Now check value counts
print(df_fip_cwp_oppm['fip_fy'].value_counts(dropna=False))

#%%
# 15. ---FIP_LIST ---
import numpy as np

# Force JCN to string so it never changes
df_fip_cwp_oppm['jcn'] = df_fip_cwp_oppm['jcn'].astype(str)

# Convert fip_fy safely to numeric (real NaN stays NaN)
df_fip_cwp_oppm['fip_fy'] = pd.to_numeric(df_fip_cwp_oppm['fip_fy'], errors='coerce')

# Create pivot table (keep NaN years as a separate column)
pivot = (
    df_fip_cwp_oppm
    .groupby(['jcn', 'fip_fy'])
    .size()
    .reset_index(name='dup_count')
    .pivot(index='jcn', columns='fip_fy', values='dup_count')
    .fillna(0)
    .reset_index()
)

# Make sure pivot jcn also stays string
pivot['jcn'] = pivot['jcn'].astype(str)

print(pivot.head(20))
#%%
# Get fiscal year columns from the pivot table (skip 'jcn' and invalid years)
fy_columns = [col for col in pivot.columns if isinstance(col, (int, float)) and col > 0]

# Create fy_list by checking each FY column value in the row
pivot['fy_list'] = pivot[fy_columns].apply(
    lambda row: ", ".join([str(int(fy)) for fy in fy_columns if row.get(fy, 0) > 0]),
    axis=1
)

# Display first 50 rows
print(pivot[['jcn', 'fy_list']].head(50))

#%%
# Ensure both 'jcn' columns are strings
df_fip_cwp_oppm['jcn'] = df_fip_cwp_oppm['jcn'].astype(str).str.strip()
pivot['jcn'] = pivot['jcn'].astype(str).str.strip()

print("Pivot jcn example:")
print(pivot[['jcn']].head(10))
print(pivot['jcn'].dtype)

print ("Dataframe jcn example")
print(df_fip_cwp_oppm[['jcn']].head(10))
print(df_fip_cwp_oppm['jcn'].dtype)

#%%
# Strip any extra whitespace
df_fip_cwp_oppm['jcn'] = df_fip_cwp_oppm['jcn'].str.strip()
pivot['jcn'] = pivot['jcn'].str.strip()

# Check which jcn values are missing in pivot
missing_jcn = set(df_fip_cwp_oppm['jcn']) - set(pivot['jcn'])
print("Example missing jcn values (up to 20):", list(missing_jcn)[:20])
print("Total missing jcn count:", len(missing_jcn))

#%%
# Ensure 'jcn' are strings and strip spaces
df_fip_cwp_oppm['jcn'] = df_fip_cwp_oppm['jcn'].astype(str).str.strip()
pivot['jcn'] = pivot['jcn'].astype(str).str.strip()

# Fill NaN in pivot fy_list just in case
pivot['fy_list'] = pivot['fy_list'].fillna('No Data')

# Create lookup dictionary from pivot
fy_lookup = pivot.set_index('jcn')['fy_list']

# Map only baseline projects
df_fip_cwp_oppm['fy_list'] = df_fip_cwp_oppm.apply(
    lambda row: fy_lookup.get(row['jcn'], 'No Data') if row['fip_baseline'] == 'YES' else None,
    axis=1
)

# Quick check
print(df_fip_cwp_oppm[['jcn', 'fip_baseline', 'fy_list']].head(20))

#%%
# Filter only baseline projects
baseline_df = df_fip_cwp_oppm[df_fip_cwp_oppm['fip_baseline'] == 'YES']

# Quick check
print(baseline_df[['jcn', 'fip_baseline', 'fy_list']].head(20))

#%%
# Convert NaN to No Data 
import pandas as pd
import numpy as np

# Replace None and NaN with 'NO DATA'
df_fip_cwp_oppm = df_fip_cwp_oppm.fillna('NO DATA')

# Replace empty strings with 'NO DATA'
df_fip_cwp_oppm.replace('', 'NO DATA', inplace=True)


print(df_fip_cwp_oppm[['jcn', 'baseline_mdce', 'baseline_sa']].head(10))

#%% 
#16.---MOVE & NO-MOVED CALCULATION ---
# count how many times each jcn appears
jcn_counts = df_fip_cwp_oppm['jcn'].value_counts()

# filter only duplicate jcns
duplicate_jcns = jcn_counts[jcn_counts > 1].index

# create a new column with None by default
df_fip_cwp_oppm['dups_occurrences'] = None

# mark first occurrence as 'first' and remaining as 'later' for duplicates
first_idx = df_fip_cwp_oppm[df_fip_cwp_oppm['jcn'].isin(duplicate_jcns)].groupby('jcn').head(1).index
df_fip_cwp_oppm.loc[first_idx, 'dups_occurrences'] = 'first'
df_fip_cwp_oppm.loc[df_fip_cwp_oppm['jcn'].isin(duplicate_jcns) & df_fip_cwp_oppm.index.isin(first_idx)==False, 'dups_occurrences'] = 'later'

# check counts
df_fip_cwp_oppm['dups_occurrences'].value_counts()

#%%
import numpy as np

# count how many times each jcn appears
jcn_counts = df_fip_cwp_oppm['jcn'].value_counts()

# identify duplicate jcns (appear more than once)
duplicate_jcns = jcn_counts[jcn_counts > 1].index

# create dups_occurrences column
df_fip_cwp_oppm['dups_occurrences'] = np.where(
    df_fip_cwp_oppm['jcn'].isin(duplicate_jcns),
    np.where(df_fip_cwp_oppm.duplicated(subset='jcn', keep='first'), 'later', 'first'),
    None
)

# create baseline_flag
# first occurrence of duplicate → baseline_cal
# unique JCNs → baseline_cal
# remaining duplicates → baseline_dup
df_fip_cwp_oppm['baseline_flag'] = np.where(
    df_fip_cwp_oppm['dups_occurrences'] == 'first', 'baseline_cal',
    np.where(df_fip_cwp_oppm['dups_occurrences'].isna(), 'baseline_cal', 'baseline_dup')
)

# check counts
print(df_fip_cwp_oppm['baseline_flag'].value_counts())

#%%
# Duplicate the baseline_flag column
dups = df_fip_cwp_oppm['baseline_flag'].copy()

# Rename the values
dups = dups.replace({'baseline_cal': 'NO MOVE', 'baseline_dup': 'MOVED FY'})

# Add the new parameter to your DataFrame
df_fip_cwp_oppm['dups'] = dups

# Check value counts of the new parameter
print(df_fip_cwp_oppm['dups'].value_counts())
#%%
# Create dups_cal_moved column
df_fip_cwp_oppm['dups_cal_moved'] = df_fip_cwp_oppm['sbit_notes'].apply(
    lambda x: 'MOVED FY' if x.upper() == 'DELAYED' else 'NO MOVE'
)

# Check how many rows are in each category
print(df_fip_cwp_oppm['dups_cal_moved'].value_counts())
#%%
#17.--- CLEAN FINAL DATASET---
# Drop columns you don't need
df_fip_cwp_oppm = df_fip_cwp_oppm.drop(columns=['fip', 'bils', 'fip_year','x&w', 'fip_jcn_description', 'baseline_flag', 'dups', 'dups_occurrences'])

# Check the DataFrame structure
df_fip_cwp_oppm.info()
#%%
# Convert the fip fy into string 
df_fip_cwp_oppm['fip_fy'] = df_fip_cwp_oppm['fip_fy'].apply(
    lambda x: 'NO DATA' if pd.isna(x) or str(x).strip().upper() == 'NO DATA' else str(int(float(x))) if str(x).replace('.', '', 1).isdigit() else str(x)
)
#%%
# Validate value counts 
df_fip_cwp_oppm['fip_fy'].value_counts()
#%%
# Convert all date column to date time 
date_cols = ['jcn_stat_dt', 'compltn_dt', 'pa_execution_dt', 
             'pa_planning_fy']

for col in date_cols:
    df_fip_cwp_oppm[col] = pd.to_datetime(df_fip_cwp_oppm[col], errors='coerce')

df_fip_cwp_oppm[['jcn_stat_dt', 'compltn_dt', 'pa_execution_dt', 
                 'pa_planning_fy' ]].dtypes
#%%
# Extract the year 
for col in date_cols:
    df_fip_cwp_oppm[col] = df_fip_cwp_oppm[col].dt.year

# No Data for NaN vaule
for col in date_cols:
    df_fip_cwp_oppm[col] = df_fip_cwp_oppm[col].fillna("NO DATA").astype(str).str.replace(".0", "", regex=False)
    
# Summary to validate
summary = {}

for col in date_cols:
    summary[col] = df_fip_cwp_oppm[col].value_counts(dropna=False).sort_index()

summary_df = pd.DataFrame(summary).fillna(0).astype(int)
print(summary_df)

#%%
df_fip_cwp_oppm['sbit_notes'].value_counts()
#%%
df_fip_cwp_oppm['fip_baseline'].value_counts()
#%%
df_fip_cwp_oppm[df_fip_cwp_oppm['fip_baseline'] == 'YES']['sbit_notes'].value_counts()
#%%
df_fip_cwp_oppm.info()
#%%
import pandas as pd
# Convert to a CSV FILE 
# Ensure all columns are strings to preserve formatting
df_clean = df_fip_cwp_oppm.astype(str)

# Replace any line breaks in all columns (important for CSV)
df_clean = df_clean.replace({r'[\r\n]': ' '}, regex=True)

# Export CSV with proper quoting to handle commas and special characters
df_clean.to_csv('df_fip_cwp_oppm_v1.csv', index=False, quoting=1)  # quoting=1 is csv.QUOTE_ALL

# Verify row count
print("Rows in DataFrame:", len(df_clean))
#%% CANCELLATIONS
#%%
#  --- OPEN & CLEAN CWP ARCHIVE JCN DELETED ---
# Read the CSV file (replace 'your_file.csv' with your actual file name)
df_cwp_deleted_cancel = pd.read_csv('p6_deleted_canceled_projects.csv')

# Display the first few rows to confirm it loaded correctly
df_cwp_deleted_cancel.info()

#%%
# Convert all column names to snake_case
df_cwp_deleted_cancel.columns = df_cwp_deleted_cancel.columns.str.strip() \
                       .str.lower() \
                       .str.replace(' ', '_') \
                       .str.replace('-', '_')
# Show the results 
df_cwp_deleted_cancel.info()

#%% 
# Remove unwanted columns 
# keep only the needed columns
df_cwp_deleted_cancel = df_cwp_deleted_cancel[[
    "project_id", 
    "jcn_status_date" 
]]

# Rename project_id -> jcn
df_cwp_deleted_cancel = df_cwp_deleted_cancel.rename(columns={"project_id": "jcn"})

# Check the result
df_cwp_deleted_cancel.info()
#%%
df_fip_cwp_oppm.shape[0]

#%%
df_cwp_deleted_cancel.shape[0]
#%%
# Perform left join
df_merged = df_fip_cwp_oppm.merge(
    df_cwp_deleted_cancel,
    on='jcn',
    how='left'
)
#%%
df_merged.shape[0]
#%%
df_merged['jcn_status_date'].value_counts()
#%%
df_merged['jcn_status_date'].notna().sum()
#%%
df_merged['jcn_status_date'].value_counts()
#%%
# Convert jcn_status_date to datetime
df_merged['jcn_status_date'] = pd.to_datetime(df_merged['jcn_status_date'])

# Extract the year (as string) while removing any '.0'
df_merged['jcn_status_year'] = df_merged['jcn_status_date'].dt.year.astype('Int64').astype('string')

# Value counts including NaN
year_counts = df_merged['jcn_status_year'].value_counts(dropna=False).sort_index()

year_counts

#%%
# Create jcn_stat_dt_cal based on the conditional logic
df_merged['jcn_stat_dt_cal'] = df_merged.apply(
    lambda row: row['jcn_status_year']
    if pd.isna(row['jcn_stat_dt']) or str(row['jcn_stat_dt']).strip() == ""
    else row['jcn_stat_dt'],
    axis=1
)

df_merged['jcn_stat_dt_cal'].value_counts(dropna=False).sort_values(ascending=False)
#%%
df_merged['sbit_status'].value_counts()

#%%
# Value count for CANCELLED, descending order
df_merged.loc[df_merged['sbit_status'] == 'CANCELLED', 'jcn_stat_dt_cal'] \
         .value_counts(dropna=False) \
         .sort_values(ascending=False)

#%%
# Filter by CANCELLED without using .loc
df_merged[df_merged['sbit_status'] == 'CANCELLED']['jcn_stat_dt_cal'] \
         .value_counts(dropna=False) \
         .sort_values(ascending=False)

#%%
import pandas as pd
# Save df_merged as a CSV file
df_merged.to_csv('df_fip_cwp_cancelations.csv', index=False)

