# cme_open_interest
## Analysis that merges CFTC COT Reports, along with Open Interest and Price data for selected commodities, to analyze large trader positions against other variables.

### Overview:
This project contains various juypter notebooks that download CFTC COT Data directly from the CFTC website, as well as CME Open Interest Files from the CME's ftp site.  Other jupyter notebooks analyze this downloaded data, showing various graphs that compare the COT reports to the price of the respective commodities, and analysis of the changes in open interest.

#### The data download notebooks are:
1. ```build_cme_open_interest_data.ipynb``` - This notebook downloads open interest history from the CME.


2. ```build_cot_data_new.ipynb``` - This notebook downloads "new format" COT reports from https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalViewable/index.htm and assembles a csv file database of that data. The notebook build_cot_data.ipynb downloads "old format" COT reports.


3. ```build_etf_shares_outstanding_data.ipynb``` - Obtain ETF shares outstanding data from various sites for Commodity ETFs (like GLD) that are used as proxies for continous commodity contracts.

#### The data analysis notebooks are:
1. ```analysis_all_new_non_fin.ipynb``` - This notebook uses ```build_cot_data_new.ipynb```, along with the other data download notebooks, to create COT/Price/OpenInterest charts for physical commodities using the CFTC's new format COT Reports.  You can use this data to track movements in large trader positions vs commodity price, etc..

2. ```analysis_all.ipynb``` - This notebook uses ```build_cot_data.ipynb```, to do the same analysis as above.

### Usage:
1. Run all the cells in:
 * build_cme_open_interest_data.ipynb
 * build_cot_data_new.ipynb (or build_cot.ipynb)
 * build_cme_open_interest_data.ipynb 
 
 
 2. Run all the cells in:
  * analysis_all_new_non_fin.ipynb when using the new format COT reports.
  * analysis_all.ipynb when using the old format COT reports.
  * *The new format reports have better detail regarder large speculative positions.*
  