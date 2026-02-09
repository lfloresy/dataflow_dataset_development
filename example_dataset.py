# -*- coding: utf-8 -*-
"""
@author: Lorenzo CTR Flores-Y
"""
#%%
# 0.INSTALL LIBRARIES & UPLOAD DATA
# Installing  cx_Oracle (intalled on anaconda_prompt)
import cx_Oracle

#%%
import os 
# Tell Python where the Oracle Instant Client is
os.add_dll_directory(r"xxxxx")

#%%
# Defina the path for the path for the oracle drivers
import os
os.environ['PATH'] = r"xxxxx" + os.environ['PATH']

#%%
# Add Oracle Instant Client folder first
os.add_dll_directory(r"xxxx")

# os.environ['PATH'] = r"xxxxxx;" + os.environ['PATH']  # optional

print("cx_Oracle version:", cx_Oracle.version)

# Connection Credentials 
dsn = cx_Oracle.makedsn("xxxx", xxxx, service_name="ndc") 
conn = cx_Oracle.connect(user="xxxxx", password="xxxxxx", dsn=dsn) #enter your user name and password to connect to the Oracle database

print("Connected successfully!")

#%%
# Create a cursor
cur = conn.cursor()

# Fetch just 10 rows to test the connection
query = """
SELECT xx, xxx, xxxx, xxx, xxx, xxx, xxx,
FROM XXXX
WHERE ROWNUM <= 10
"""
cur.execute(query)

# Fetch the rows
rows = cur.fetchall()

# Print them to confirm
for row in rows:
    print(row)

#%%
# --- 1.xxx TABALE DEVELOPMENT ---
# Import to a pandas dataframe 
import pandas as pd

query = """
SELECT xx, xxx, xxxx, xxx, xxx, xxx,
FROM xxxx
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
# Filter x_CIPs
# Create the list of cips 
list = [
    "F01", "F01", "F06", "F11", "F12",
   ]

# Keep only the x_CIPs
df_xxx = df[df["cip"].isin(list)]

# Show the results
df_xxx.info()

print(df_xxx.shape)
#%%
# Remove status N, U, W, X
df_xxx = df_xxx[~df_xxx["status"].isin(["N", "U", "W", "X"])]

# Show all unique values to confirm
print(df_xxx["status"].unique())

#%%
# Count null values in compltn_dt
null_count = df_xxx["compltn_dt"].isnull().sum()
print("Number of null compltn_dt values:", null_count)

#%% 
# Convert compltn_dt to datetime (empty strings become NaT)
df_xxx["compltn_dt"] = pd.to_datetime(df_xxx["compltn_dt"], errors="coerce")

# Create date_flag with 3 categories
def date_category(x):
    if pd.isna(x):
        return "EMPTY"
    elif x > pd.Timestamp("2016-09-30"):
        return "IN"
    else:
        return "OUT"

df_xxx["date_flag"] = df_xxx["compltn_dt"].apply(date_category)

# Count the occurrences of each category
date_flag_counts = df_xxx["date_flag"].value_counts()
print(date_flag_counts)

#%%
# Keep only rows where date_flag is not OUT
df_xxx_filtered = df_xxx[df_xxx["date_flag"] != "OUT"]

# Check the counts of remaining categories
print(df_xxx_filtered["date_flag"].value_counts())

#%%
# Define the completed statuses
completed_statuses = ["X", "X", "X, "X", "X"]

# Create a new column empty_flag
df_xxx_filtered["empty_flag"] = df_xxx_filtered.apply(
    lambda row: "YES" if row["date_flag"] == "EMPTY" and row["status"] in completed_statuses else "NO",
    axis=1
)

# Ccheck the counts
print(df_xxx_filtered["empty_flag"].value_counts())
#%%
#  Exclude rows where empty_flag is "YES"
df_xxx_final = df_xxx_filtered[df_xxx_filtered["empty_flag"] != "YES"]

#  Drop the helper columns date_flag and empty_flag
df_xxx_final = df_xxx_final.drop(columns=["date_flag", "empty_flag"])

# Check the final DataFrame
print("Final number of rows:", len(df_xxx_final))

print(df_xxx_final['xxx_status'].value_counts())
#%%
# --- 2.LOAD xx17-xx27 FILE ---
import os

# Check current working directory
print("Current working directory:", os.getcwd())

#%%
# Read the CSV file (replace 'your_file.csv' with your actual file name)
df_x_xx = pd.read_csv('xx17-xx27_AUG2025.csv')

# Display the first few rows to confirm it loaded correctly
df_x_xx.info()
#%%
# Convert all column names to snake_case
df_x_xx.columns = df_x_xx.columns.str.strip() \
                       .str.lower() \
                       .str.replace(' ', '_') \
                       .str.replace('-', '_')
# Show the results 
df_x_xx.info()
print(len(df_x_xx))

#%%
# Create a duplicate of df_x_xx
df_xx_v2 = df_x_xx.copy()

df_xx_v2.info()

#%% Clean the file
# List of columns to remove
cols_to_drop = [
    "xxx", "xxxx", "xxxx", "xxx", "xx",
   ]

# Drop the columns
df_x_xx = df_x_xx.drop(columns=cols_to_drop, errors="ignore")

# Verixx result
df_x_xx.info()

#%%
# --- 3.JOIN xxx & Fxx17-xx27 FOR ADDITIONS ---

# Perform a left join first
merged = pd.merge(
    df_xxx_final,
    df_x_xx,
    on="xxx",
    how="left",
    indicator=True
)

# Keep only rows from df_xxx_final that did NOT match in df_x_xx
df_xxx_add = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

# Show the results 
df_xxx_add.info()

# Check the results
print("Rows in df_xxx_final:", len(df_xxx_final))
print("Rows in df_x_xx:", len(df_x_xx))
print("Rows in df_xxx_add (unmatched):", len(df_xxx_add))

print(df_xxx_add['xxx_status'].value_counts())

df_xxx_add.info()

#%%
# --- 4. JOIN ADDITIONS WITH SHORTFALLS ---

# Read the CSV file (replace 'your_file.csv' with your actual file name)
df_shortfalls = pd.read_csv('x_shortfalls.csv')

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
# Remove all the parametes and only keep xxx
df_shortfalls = df_shortfalls[['xxx']]

# Add new column with constant value
df_shortfalls['x_add_shortfall'] = "YES"

print(df_shortfalls.head())
print("Rows in df_shortfalls:", len(df_shortfalls))
#%%
# Perform a left join addition & shortfalls 
df_xxx_add_shortfalls = pd.merge(
    df_xxx_add,
    df_shortfalls,
    on="xxx",
    how="left"
)

# Verixx row counts
print("Rows in df_xxx_add:", len(df_xxx_add))
print("Rows in df_shortfalls:", len(df_shortfalls))
print("Rows in df_xxx_add_shortfalls (left join result):", len(df_xxx_add_shortfalls))

print(df_xxx_add_shortfalls['xxx_status'].value_counts())

df_xxx_add_shortfalls.info()
#%%
# Add x_baseline column
df_x_xx["x_baseline"] = "YES"
df_xxx_add_shortfalls["x_baseline"] = "NO"

# Force multi_year = "NO" for all xxx_add_shortfalls
df_xxx_add_shortfalls["multi_year"] = "NO"
#%%
df_xxx_add_shortfalls.info()
#%%
# Keep only the neded parameters
df_xxx_subset = df[["xxx", "xxx_status", "most_detailed_cost_est", "xxx_stat_dt" ]]

df_xxx_subset.info()

print(len(df_xxx_subset))
print(df_xxx_subset['xxx_status'].value_counts())

df_xxx_subset.info()

#%% df_x_xxx_subset_updated
# Left join with df_x_xx as the left table
df_x_xxx_subset_updated = df_xx_v2_subset.merge(
    df_xxx_subset,
    on="xxx",
    how="left",
    suffixes=("", "_xxx")
)

# Validate with row counts
print("Row count df_x_xx:", len(df_x_xx))
print("Row count df_xxx_subset:", len(df_xxx_subset))
print("Row count df_x_xxx_subset_updated:", len(df_x_xxx_subset_updated))

#%% UPPDATED df_x_xxx_subset
# Left join with df_x_xx as the left table
df_x_xxx_subset_updated = df_xx_v2_subset.merge(
    df_xxx_subset,
    on="xxx",
    how="left",
    suffixes=("", "_xxx")
)

# Validate with row counts
print("Row count df_x_xx_v2_subset:", len(df_xx_v2_subset))
print("Row count df_xxx_subset:", len(df_xxx_subset))
print("Row count df_x_xxx_subset_updated:", len(df_x_xxx_subset_updated))


#%%
#%% ORIGINAL df_x_xxx_subset
# Left join with df_x_xx as the left table
df_x_xxx_subset = df_x_xx.merge(
    df_xxx_subset,
    on="xxx",
    how="left",
    suffixes=("", "_xxx")
)

# Validate with row counts
print("Row count df_x_xx:", len(df_x_xx))
print("Row count df_xxx_subset:", len(df_xxx_subset))
print("Row count df_x_xxx_subset:", len(df_x_xxx_subset))

print(df_x_xxx_subset['xxx_status'].value_counts())

#%%
# --- 5.UNION x_xx & xxx_ADD ---

# Union (row-wise append)
df_union_x_add = pd.concat([df_xxx_add_shortfalls, df_x_xxx_subset_updated], ignore_index=True)

# Check the result
print(len(df_xxx_add_shortfalls))
print(len(df_x_xx))              # should be 9510
print(len(df_union_x_add))               

print(df_union_x_add['xxx_status'].value_counts(dropna=False))
#%%
# Validate the union 
df_union_x_add.info()

#%%
# Count unique xxx values
unique_xxx_count = df_union_x_add["xxx"].nunique()
print("Unique xxx count:", unique_xxx_count)

# Count total xxx rows
total_xxx_rows = df_union_x_add["xxx"].count()
print("Total xxx rows:", total_xxx_rows)

# Distribution of multi_year
print("\nmulti_year counts:")
print(df_union_x_add["multi_year"].value_counts(dropna=False))

# Distribution of x_baseline
print("\nx_baseline counts:")
print(df_union_x_add["x_baseline"].value_counts(dropna=False))
#%%
# Shortfalls 
# Replace any NaN with NO (safety check)
df_union_x_add["x_add_shortfall"] = df_union_x_add["x_add_shortfall"].fillna("NO")
print(df_union_x_add["x_add_shortfall"].value_counts(dropna=False))
#%% 
# --- 6.GET OPPM CHANGE REASONS & MULTI-YEAR --- 

# Read the CSV file (replace 'your_file.csv' with your actual file name)
df_my = pd.read_csv('my_xx17_xx28_v1.csv')

df_my = df_my[["xxx", "MULTI_YEAR", "x_xx"]]
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
# Sort by xxx and x_xx descending
df_my = df_my.sort_values(['xxx', 'xxx'], ascending=[True, False])

# Keep only the latest x_xx per xxx
df_my = df_my.groupby('xxx', group_keys=False).head(1)

# Validate
print("Number of rows:", len(df_my))
print(df_my['xxx'].value_counts())  # all values should be 1

df_my.info()
#%%
# Drop x_xx
df_my = df_my.drop(columns=['x_xx'])

df_my.info()

#%%
# Get the Change Reasons dataset and format it 
df_change_reasons = pd.read_csv('change_tracking_sep.csv')

# Convert all column names to snake_case
df_change_reasons.columns = df_change_reasons.columns.str.lower()

# Select only the parameters needed for the analysis 
df_change_reasons = df_change_reasons[["xxx", "xxx", "xxx", "xxx", "xxx"]]

df_change_reasons.info()

#%%
# Exclude all the null values
df_change_reasons = df_change_reasons[df_change_reasons['xxx'].notna()]

print("Number of rows:", len(df_change_reasons))


# Rename items to x_xxx_description 
df_change_reasons = df_change_reasons.rename(columns={'items': 'x_xxx_description'})

df_change_reasons.info()

#%%
# Delete duplicates 
df_change_reasons = df_change_reasons.drop_duplicates(subset='xxx', keep='first')

print("Number of rows:", len(df_change_reasons))

#%%
df_my_change_reasons = pd.merge(
    df_my,
    df_change_reasons,
    on='xxx',
    how='outer'
)

#%%
# 7. --- JOIN OPPM CHANGE REASONS - MULTI-YEAR WITH x --- 

#Validating the correct table
print(df_union_x_add['x_baseline'].value_counts())

# Validting the other table 
print(len(df_union_x_add))


# Validting the other table 
print(len(df_my_change_reasons))

#%%
df_x_xxx_oppm = df_union_x_add.merge(
    df_my_change_reasons,
    on='xxx',
    how='left'
)

df_x_xxx_oppm.info()
#%%
# Drop multi_year_x 
df_x_xxx_oppm = df_x_xxx_oppm.drop(columns='multi_year_x')

# Rename mulit year and fill all the non values for NO
df_x_xxx_oppm['multi_year'] = df_x_xxx_oppm['multi_year_y'].fillna('NO')

df_x_xxx_oppm = df_x_xxx_oppm.drop(columns='multi_year_y')

# Validate the changes 
df_x_xxx_oppm.info()

print(df_x_xxx_oppm['multi_year'].value_counts())

print(df_x_xxx_oppm['x_baseline'].value_counts())


#%%
# Validate the table 
print(df_x_xxx_oppm['xxx_status'].value_counts(dropna=False))

print(df_x_xxx_oppm['x_baseline'].value_counts(dropna=False))


print(
    df_x_xxx_oppm.pivot_table(
        index='xxx',
        columns='xxx',
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
df_x_xxx_oppm["sbit_status"] = df_x_xxx_oppm["xxx_status"].map(status_mapping)


# Make the NaN values Canceled 
df_x_xxx_oppm["sbit_status"] = (
    df_x_xxx_oppm["xxx_status"]
    .map(status_mapping)
    .fillna("CANCELLED")
)

# Validate the SBIT_STATUS 
print(df_x_xxx_oppm['sbit_status'].value_counts(dropna=False))

#%%
# 9. ---CIP CALCULATED  ----
df_x_xxx_oppm['cip'] = df_x_xxx_oppm['cip'].combine_first(df_x_xxx_oppm['x_cip'])

print(df_x_xxx_oppm['cip'].value_counts(dropna=False))

df_x_xxx_oppm.info()
#%%
#10. --- RANK---
df_x_xxx_oppm['row_number'] = (
    df_x_xxx_oppm
    .sort_values(['xxx', 'x_year'], ascending=[True, False])  # sort by xxx, then x_year descending
    .groupby('xxx')
    .cumcount() + 1  # row_number starting at 1
)

print(df_x_xxx_oppm['row_number'].value_counts(dropna=False))
#%% 

df_x_xxx_oppm['cip_category'] = df_x_xxx_oppm['cip'].apply(
    lambda x: 'IIJA' if x in iija_values else 'LEGACY'
)
print(df_x_xxx_oppm['cip_category'].value_counts(dropna=False))

#%%
#11. --- SBIT NOTES --- 
# x Column 

# Create new column 'x' based on parameter 'cip'
df_x_xxx_oppm['x'] = df_x_xxx_oppm['cip'].apply(
    lambda x: 'in' if str(x).lower() in cip_list else 'out'
)
print(df_x_xxx_oppm['x'].value_counts(dropna=False))

# Create new colunn 'bils' based on parameter 'cip'

# Create new column 'bils' based on parameter 'cip'
df_x_xxx_oppm['bils'] = df_x_xxx_oppm['cip'].apply(
    lambda x: 'in' if str(x).lower() in bils_list else 'out'
)
print(df_x_xxx_oppm['bils'].value_counts(dropna=False))

# Create new column x&w
# values considered as 'in' for x&w
xw_list = ['x', 'w', 'u']

# create new column 'x&w' based on column 'xxx_status-1'
# values considered as 'in' for x&w
xw_list = ['x', 'w', 'u']

# create new column 'x&w' including NaN as 'in'
df_x_xxx_oppm['x&w'] = df_x_xxx_oppm['xxx_status'].apply(
    lambda x: 'in' if (pd.isna(x) or str(x).lower() in xw_list) else 'out'
)

print(df_x_xxx_oppm['x&w'].value_counts(dropna=False))
#%%
import numpy as np
import pandas as pd

# ensure pa_execution_dt is datetime
df_x_xxx_oppm['pa_execution_dt'] = pd.to_datetime(df_x_xxx_oppm['pa_execution_dt'], errors='coerce')

#%%
# define conditions
conditions = [
    # CANCELLED
    (df_x_xxx_oppm['x&w'] == 'in'),
    
    # MOVED TO IIJA
    (df_x_xxx_oppm['bils'] == 'in') & (df_x_xxx_oppm['x&w'] == 'out'),
    
    # MOVED TO OTHER CIP
    (df_x_xxx_oppm['x'] == 'out') & (df_x_xxx_oppm['bils'] == 'out') & (df_x_xxx_oppm['x&w'] == 'out'),
    
    # DELAYED by year rules
    (df_x_xxx_oppm['x&w'] == 'out') & (df_x_xxx_oppm['pa_planning_xx'] == '2017') & (df_x_xxx_oppm['pa_execution_dt'] > '2018-03-31'),
    (df_x_xxx_oppm['x&w'] == 'out') & (df_x_xxx_oppm['pa_planning_xx'] == '2018') & (df_x_xxx_oppm['pa_execution_dt'] > '2019-03-31'),
    (df_x_xxx_oppm['x&w'] == 'out') & (df_x_xxx_oppm['pa_planning_xx'] == '2019') & (df_x_xxx_oppm['pa_execution_dt'] > '2020-03-31'),
    (df_x_xxx_oppm['x&w'] == 'out') & (df_x_xxx_oppm['pa_planning_xx'] == '2020') & (df_x_xxx_oppm['pa_execution_dt'] > '2021-03-31'),
    (df_x_xxx_oppm['x&w'] == 'out') & (df_x_xxx_oppm['pa_planning_xx'] == '2021') & (df_x_xxx_oppm['pa_execution_dt'] > '2022-03-31'),
    (df_x_xxx_oppm['x&w'] == 'out') & (df_x_xxx_oppm['pa_planning_xx'] == '2022') & (df_x_xxx_oppm['pa_execution_dt'] > '2023-03-31'),
    (df_x_xxx_oppm['x&w'] == 'out') & (df_x_xxx_oppm['pa_planning_xx'] == '2023') & (df_x_xxx_oppm['pa_execution_dt'] > '2024-03-31'),
    (df_x_xxx_oppm['x&w'] == 'out') & (df_x_xxx_oppm['pa_planning_xx'] == '2024') & (df_x_xxx_oppm['pa_execution_dt'] > '2025-03-31'),
    (df_x_xxx_oppm['x&w'] == 'out') & (df_x_xxx_oppm['pa_planning_xx'] == '2025') & (df_x_xxx_oppm['pa_execution_dt'] > '2026-03-31'),
    (df_x_xxx_oppm['x&w'] == 'out') & (df_x_xxx_oppm['pa_planning_xx'] == '2026') & (df_x_xxx_oppm['pa_execution_dt'] > '2027-03-31'),
    
    # NULL DELAYED
    (df_x_xxx_oppm['pa_execution_dt'].isna())
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
df_x_xxx_oppm['sbit_notes'] = np.select(conditions, choices, default="NO CHANGE")

print(df_x_xxx_oppm['sbit_notes'].value_counts(dropna=False))
print(df_x_xxx_oppm['sbit_status'].value_counts(dropna=False))

#%%
#12. --- x_ADDITION_CAT ---
import numpy as np
import pandas as pd

# Ensure text columns are strings
df_x_xxx_oppm['xxx_desc'] = df_x_xxx_oppm['xxx_desc'].astype(str)
df_x_xxx_oppm['proj_cd'] = df_x_xxx_oppm['proj_cd'].astype(str)

# Define conditions
conditions = [
    # CAB_GLASS
    df_x_xxx_oppm['xxx_desc'].str.contains('CAB GLASS', case=False, na=False) |
    df_x_xxx_oppm['xxx_desc'].str.contains('WINDOW PANELS', case=False, na=False) |
    df_x_xxx_oppm['xxx_desc'].str.contains('WINDOW GLASS', case=False, na=False) |
    df_x_xxx_oppm['xxx_desc'].str.contains('WINDOW PANEL', case=False, na=False) |
    df_x_xxx_oppm['xxx_desc'].str.contains('CAB PANEL', case=False, na=False) |
    df_x_xxx_oppm['xxx_desc'].str.contains('CAB PANELS', case=False, na=False),
    
    # CAB_SHADE
    df_x_xxx_oppm['xxx_desc'].str.contains('CAB SHADE', case=False, na=False) |
    df_x_xxx_oppm['xxx_desc'].str.contains('CAB SHADES', case=False, na=False) |
    df_x_xxx_oppm['xxx_desc'].str.contains('WINDOW SHADE', case=False, na=False) |
    df_x_xxx_oppm['xxx_desc'].str.contains('WINDOW SHADES', case=False, na=False),
    
    # ERMS
    df_x_xxx_oppm['proj_cd'].isin(['98310684', '9831J684']),
    
    # BATTERIES
    df_x_xxx_oppm['proj_cd'].isin(['98310686', '98310688', '98310695', '9831J695', '9831J696'])
]

# Corresponding choices
choices = [
    'CAB_GLASS',
    'CAB_SHADE',
    'ERMS',
    'BATTERIES'
]

# Apply conditions to create new column
df_x_xxx_oppm['x_additions_cat'] = np.select(conditions, choices, default='NO_CAT')

# Validate 
print(df_x_xxx_oppm['x_additions_cat'].value_counts(dropna=False))
#%%
# 13. --- DUPlICATION COUNTS --- 
# Count duplicates of each xxx
df_x_xxx_oppm['dup_counts'] = df_x_xxx_oppm.groupby('xxx')['xxx'].transform('count')

# Validate 
print(df_x_xxx_oppm['dup_counts'].value_counts(dropna=False))

df_x_xxx_oppm.info()
#%%
#14.---x_INSTANCE ---
# Create xxx_instances based on dup_counts
df_x_xxx_oppm['xxx_instances'] = df_x_xxx_oppm['dup_counts'].apply(
    lambda x: 'DUPLICATE' if x > 1 else 'UNIQUE'
)

# Validate 
print(df_x_xxx_oppm['xxx_instances'].value_counts(dropna=False))

#%%
df_x_xxx_oppm.info()
print(df_x_xxx_oppm['x_xx'].value_counts(dropna=False))

print(
    df_x_xxx_oppm[df_x_xxx_oppm['x_baseline'] == 'YES']['x_year']
    .value_counts(dropna=False)
)
#%%
# Convert x_xx to numeric (ignores errors, keeps NaN), then to Int where possible, then to string
df_x_xxx_oppm['x_xx'] = pd.to_numeric(df_x_xxx_oppm['x_xx'], errors='coerce')
df_x_xxx_oppm['x_xx'] = df_x_xxx_oppm['x_xx'].apply(lambda x: str(int(x)) if pd.notna(x) else 'NaN')

# Now check value counts
print(df_x_xxx_oppm['x_xx'].value_counts(dropna=False))

#%%
# 15. ---x_LIST ---
import numpy as np

# Force xxx to string so it never changes
df_x_xxx_oppm['xxx'] = df_x_xxx_oppm['xxx'].astype(str)

# Convert x_xx safely to numeric (real NaN stays NaN)
df_x_xxx_oppm['x_xx'] = pd.to_numeric(df_x_xxx_oppm['x_xx'], errors='coerce')

# Create pivot table (keep NaN years as a separate column)
pivot = (
    df_x_xxx_oppm
    .groupby(['xxx', 'x_xx'])
    .size()
    .reset_index(name='dup_count')
    .pivot(index='xxx', columns='x_xx', values='dup_count')
    .fillna(0)
    .reset_index()
)

# Make sure pivot xxx also stays string
pivot['xxx'] = pivot['xxx'].astype(str)

print(pivot.head(20))
#%%
# Get fiscal year columns from the pivot table (skip 'xxx' and invalid years)
xx_columns = [col for col in pivot.columns if isinstance(col, (int, float)) and col > 0]

# Create xx_list by checking each xx column value in the row
pivot['xx_list'] = pivot[xx_columns].apply(
    lambda row: ", ".join([str(int(xx)) for xx in xx_columns if row.get(xx, 0) > 0]),
    axis=1
)

# Display first 50 rows
print(pivot[['xxx', 'xx_list']].head(50))

#%%
# Ensure both 'xxx' columns are strings
df_x_xxx_oppm['xxx'] = df_x_xxx_oppm['xxx'].astype(str).str.strip()
pivot['xxx'] = pivot['xxx'].astype(str).str.strip()

print("Pivot xxx example:")
print(pivot[['xxx']].head(10))
print(pivot['xxx'].dtype)

print ("Dataframe xxx example")
print(df_x_xxx_oppm[['xxx']].head(10))
print(df_x_xxx_oppm['xxx'].dtype)

#%%
# Strip any extra whitespace
df_x_xxx_oppm['xxx'] = df_x_xxx_oppm['xxx'].str.strip()
pivot['xxx'] = pivot['xxx'].str.strip()

# Check which xxx values are missing in pivot
missing_xxx = set(df_x_xxx_oppm['xxx']) - set(pivot['xxx'])
print("Example missing xxx values (up to 20):", list(missing_xxx)[:20])
print("Total missing xxx count:", len(missing_xxx))

#%%
# Ensure 'xxx' are strings and strip spaces
df_x_xxx_oppm['xxx'] = df_x_xxx_oppm['xxx'].astype(str).str.strip()
pivot['xxx'] = pivot['xxx'].astype(str).str.strip()

# Fill NaN in pivot xx_list just in case
pivot['xx_list'] = pivot['xx_list'].fillna('No Data')

# Create lookup dictionary from pivot
xx_lookup = pivot.set_index('xxx')['xx_list']

# Map only baseline projects
df_x_xxx_oppm['xx_list'] = df_x_xxx_oppm.apply(
    lambda row: xx_lookup.get(row['xxx'], 'No Data') if row['x_baseline'] == 'YES' else None,
    axis=1
)

# Quick check
print(df_x_xxx_oppm[['xxx', 'x_baseline', 'xx_list']].head(20))

#%%
# Filter only baseline projects
baseline_df = df_x_xxx_oppm[df_x_xxx_oppm['x_baseline'] == 'YES']

# Quick check
print(baseline_df[['xxx', 'x_baseline', 'xx_list']].head(20))

#%%
# Convert NaN to No Data 
import pandas as pd
import numpy as np

# Replace None and NaN with 'NO DATA'
df_x_xxx_oppm = df_x_xxx_oppm.fillna('NO DATA')

# Replace empty strings with 'NO DATA'
df_x_xxx_oppm.replace('', 'NO DATA', inplace=True)


print(df_x_xxx_oppm[['xxx', 'baseline_mdce', 'baseline_sa']].head(10))

#%% 
#16.---MOVE & NO-MOVED CALCULATION ---
# count how many times each xxx appears
xxx_counts = df_x_xxx_oppm['xxx'].value_counts()

# filter only duplicate xxxs
duplicate_xxxs = xxx_counts[xxx_counts > 1].index

# create a new column with None by default
df_x_xxx_oppm['dups_occurrences'] = None

# mark first occurrence as 'first' and remaining as 'later' for duplicates
first_idx = df_x_xxx_oppm[df_x_xxx_oppm['xxx'].isin(duplicate_xxxs)].groupby('xxx').head(1).index
df_x_xxx_oppm.loc[first_idx, 'dups_occurrences'] = 'first'
df_x_xxx_oppm.loc[df_x_xxx_oppm['xxx'].isin(duplicate_xxxs) & df_x_xxx_oppm.index.isin(first_idx)==False, 'dups_occurrences'] = 'later'

# check counts
df_x_xxx_oppm['dups_occurrences'].value_counts()

#%%
import numpy as np

# count how many times each xxx appears
xxx_counts = df_x_xxx_oppm['xxx'].value_counts()

# identixx duplicate xxxs (appear more than once)
duplicate_xxxs = xxx_counts[xxx_counts > 1].index

# create dups_occurrences column
df_x_xxx_oppm['dups_occurrences'] = np.where(
    df_x_xxx_oppm['xxx'].isin(duplicate_xxxs),
    np.where(df_x_xxx_oppm.duplicated(subset='xxx', keep='first'), 'later', 'first'),
    None
)

# create baseline_flag
# first occurrence of duplicate → baseline_cal
# unique xxxs → baseline_cal
# remaining duplicates → baseline_dup
df_x_xxx_oppm['baseline_flag'] = np.where(
    df_x_xxx_oppm['dups_occurrences'] == 'first', 'baseline_cal',
    np.where(df_x_xxx_oppm['dups_occurrences'].isna(), 'baseline_cal', 'baseline_dup')
)

# check counts
print(df_x_xxx_oppm['baseline_flag'].value_counts())

#%%
# Duplicate the baseline_flag column
dups = df_x_xxx_oppm['baseline_flag'].copy()

# Rename the values
dups = dups.replace({'baseline_cal': 'NO MOVE', 'baseline_dup': 'MOVED xx'})

# Add the new parameter to your DataFrame
df_x_xxx_oppm['dups'] = dups

# Check value counts of the new parameter
print(df_x_xxx_oppm['dups'].value_counts())
#%%
# Create dups_cal_moved column
df_x_xxx_oppm['dups_cal_moved'] = df_x_xxx_oppm['sbit_notes'].apply(
    lambda x: 'MOVED xx' if x.upper() == 'DELAYED' else 'NO MOVE'
)

# Check how many rows are in each category
print(df_x_xxx_oppm['dups_cal_moved'].value_counts())
#%%
#17.--- CLEAN FINAL DATASET---
# Drop columns you don't need
df_x_xxx_oppm = df_x_xxx_oppm.drop(columns=['x', 'bils', 'x_year','x&w', 'x_xxx_description', 'baseline_flag', 'dups', 'dups_occurrences'])

# Check the DataFrame structure
df_x_xxx_oppm.info()
#%%
# Convert the x xx into string 
df_x_xxx_oppm['x_xx'] = df_x_xxx_oppm['x_xx'].apply(
    lambda x: 'NO DATA' if pd.isna(x) or str(x).strip().upper() == 'NO DATA' else str(int(float(x))) if str(x).replace('.', '', 1).isdigit() else str(x)
)
#%%
# Validate value counts 
df_x_xxx_oppm['x_xx'].value_counts()
#%%
# Convert all date column to date time 
date_cols = ['xxx_stat_dt', 'compltn_dt', 'pa_execution_dt', 
             'pa_planning_xx']

for col in date_cols:
    df_x_xxx_oppm[col] = pd.to_datetime(df_x_xxx_oppm[col], errors='coerce')

df_x_xxx_oppm[['xxx_stat_dt', 'compltn_dt', 'pa_execution_dt', 
                 'pa_planning_xx' ]].dtypes
#%%
# Extract the year 
for col in date_cols:
    df_x_xxx_oppm[col] = df_x_xxx_oppm[col].dt.year

# No Data for NaN vaule
for col in date_cols:
    df_x_xxx_oppm[col] = df_x_xxx_oppm[col].fillna("NO DATA").astype(str).str.replace(".0", "", regex=False)
    
# Summary to validate
summary = {}

for col in date_cols:
    summary[col] = df_x_xxx_oppm[col].value_counts(dropna=False).sort_index()

summary_df = pd.DataFrame(summary).fillna(0).astype(int)
print(summary_df)

#%%
df_x_xxx_oppm['sbit_notes'].value_counts()
#%%
df_x_xxx_oppm['x_baseline'].value_counts()
#%%
df_x_xxx_oppm[df_x_xxx_oppm['x_baseline'] == 'YES']['sbit_notes'].value_counts()
#%%
df_x_xxx_oppm.info()
#%%
import pandas as pd
# Convert to a CSV FILE 
# Ensure all columns are strings to preserve formatting
df_clean = df_x_xxx_oppm.astype(str)

# Replace any line breaks in all columns (important for CSV)
df_clean = df_clean.replace({r'[\r\n]': ' '}, regex=True)

# Export CSV with proper quoting to handle commas and special characters
df_clean.to_csv('df_x_xxx_oppm_v1.csv', index=False, quoting=1)  # quoting=1 is csv.QUOTE_ALL

# Verixx row count
print("Rows in DataFrame:", len(df_clean))
#%% CANCELLATIONS
#%%
#  --- OPEN & CLEAN xxx ARCHIVE xxx DELETED ---
# Read the CSV file (replace 'your_file.csv' with your actual file name)
df_xxx_deleted_cancel = pd.read_csv('p6_deleted_canceled_projects.csv')

# Display the first few rows to confirm it loaded correctly
df_xxx_deleted_cancel.info()

#%%
# Convert all column names to snake_case
df_xxx_deleted_cancel.columns = df_xxx_deleted_cancel.columns.str.strip() \
                       .str.lower() \
                       .str.replace(' ', '_') \
                       .str.replace('-', '_')
# Show the results 
df_xxx_deleted_cancel.info()

#%% 
# Remove unwanted columns 
# keep only the needed columns
df_xxx_deleted_cancel = df_xxx_deleted_cancel[[
    "project_id", 
    "xxx_status_date" 
]]

# Rename project_id -> xxx
df_xxx_deleted_cancel = df_xxx_deleted_cancel.rename(columns={"project_id": "xxx"})

# Check the result
df_xxx_deleted_cancel.info()
#%%
df_x_xxx_oppm.shape[0]

#%%
df_xxx_deleted_cancel.shape[0]
#%%
# Perform left join
df_merged = df_x_xxx_oppm.merge(
    df_xxx_deleted_cancel,
    on='xxx',
    how='left'
)
#%%
df_merged.shape[0]
#%%
df_merged['xxx_status_date'].value_counts()
#%%
df_merged['xxx_status_date'].notna().sum()
#%%
df_merged['xxx_status_date'].value_counts()
#%%
# Convert xxx_status_date to datetime
df_merged['xxx_status_date'] = pd.to_datetime(df_merged['xxx_status_date'])

# Extract the year (as string) while removing any '.0'
df_merged['xxx_status_year'] = df_merged['xxx_status_date'].dt.year.astype('Int64').astype('string')

# Value counts including NaN
year_counts = df_merged['xxx_status_year'].value_counts(dropna=False).sort_index()

year_counts

#%%
# Create xxx_stat_dt_cal based on the conditional logic
df_merged['xxx_stat_dt_cal'] = df_merged.apply(
    lambda row: row['xxx_status_year']
    if pd.isna(row['xxx_stat_dt']) or str(row['xxx_stat_dt']).strip() == ""
    else row['xxx_stat_dt'],
    axis=1
)

df_merged['xxx_stat_dt_cal'].value_counts(dropna=False).sort_values(ascending=False)
#%%
df_merged['sbit_status'].value_counts()

#%%
# Value count for CANCELLED, descending order
df_merged.loc[df_merged['sbit_status'] == 'CANCELLED', 'xxx_stat_dt_cal'] \
         .value_counts(dropna=False) \
         .sort_values(ascending=False)

#%%
# Filter by CANCELLED without using .loc
df_merged[df_merged['sbit_status'] == 'CANCELLED']['xxx_stat_dt_cal'] \
         .value_counts(dropna=False) \
         .sort_values(ascending=False)

#%%
import pandas as pd
# Save df_merged as a CSV file
df_merged.to_csv('df_x_xxx_cancelations.csv', index=False)






