**Date - 20/9/2025**

At the moment the backtesting algorithm for the program only works by calculating and mapping the return from the trading algorithm using `matplotlib`. 

### Data Input
Previously the user had to make sure that the data to back test on was in the correct format in the correct folder within the project directory, however now I've change the program to identify all files and folders within the `backtestingData` folder and then at run time allowing the user to chose which folder of data or single data file they want to back test on. To implement this I had to implement the `inquirer` module which provides the user with a selection list about what data they wanted to use for back testing. Once the user has then selected a file or folder to test from the program then identifies if it is just a single file they've selected, or if there is a whole folder that needs to be read then formats the data into a list of data frames to be backtested, the snippet for this can be seen below:

```python
def getDfArr():

	folder_choice = [inquirer.List('selection',
		message='Select A Folder/File to use for the Backtesting Data:',
		choices=getDataFolderList())
		]
		
	folder_to_use = inquirer.prompt(folder_choice)['selection']
	split = folder_to_use.split('.')
	
	if len(split) == 1: #if the selection is a folder
		print('reading data from a folder')
		df_arr = readDataFolder(split[0])
	
	else: #if the selection is a single csv file with backtesting data
		print('reading data from a single folder')
		df_arr = readFile(folder_to_use)
	
	return df_arr
```

### Data Metrics/Performance Analysis
The next change was to use *sharpe ratio* and *sortino ratio* to evaluate performance as currently the model just used the profitability of the model which is effective when comparing across multiple different approaches and then these 2 metrics are returned in a dictionary to help with data organisation of the algorithm. The snippet for this section of the program can be seen below:
```python
def getMetrics(tracking_df):

	metrics_dict = {}
	periodic_returns = tracking_df['portfolio_val'].pct_change().dropna()
	
	sharpe_avg_return = periodic_returns.mean()
	sharpe_std = periodic_returns.std()
	daily_sharpe_ratio = sharpe_avg_return / sharpe_std
	annualised_sharpe_ratio = daily_sharpe_ratio * np.sqrt(252)
	metrics_dict['sharpe_ratio'] = annualised_sharpe_ratio
	
	sortino_avg_return = periodic_returns.mean()
	target_return = 0 #any level of return below this is used for sortino
	downside_returns = periodic_returns.copy()
	downside_returns[downside_returns > target_return] = 0
	sortino_std = downside_returns.std()
	
	if sortino_std == 0:
		daily_sortino_ratio = np.inf
	
	else:
		daily_sortino_ratio = (sortino_avg_return - target_return) / sortino_std
	
	annualised_sortino_ratio = daily_sortino_ratio * np.sqrt(252)
	metrics_dict['sortino_ratio'] = annualised_sortino_ratio
	
	return metrics_dict
```

