Since the last update - [[24-9-2025 - Variable vs Fixed Volatility]] - I've now built out a program that can identify **n** hidden states within a given dataset, this code uses just numpy and pandas to identify the hidden states and then provides the **mean** (mu) and the **volatility** (sigma) of each state. Then once we have these metrics you can calculate the likelihood - and hence probability - of which state a given value would be in.

This acts as the basis for the **MSGARCH** model because we can now detect the states for the **Markov Switching** section of the algorithm. Well, i say this is the code for the **MS** section when in reality the code I've produced so far can only identify states in independent data points, for example predicting if someone with a given height is male or female - whereas what I need for my model is to be able to take a given number of dependent variables from the input data, because markets carry momentum each days' returns are not wholly separate from one another so this is the next logical step, then its *just* a matter of implementing a GARCH fitting model at the same time so the 2 models contextualise to one-another.

# The Code
The program hinges on an expectation maximisation algorithm which tunes the parameters of the model to give us each states parameters. However the model doesn't start with the the EM algorithm, this is because the algorithm needs to take in a previous mean and standard deviation value but to get the mean and standard deviation we need the EM algorithm to predict them for each state which results in a *what came first the chicken or the egg* type scenario.

## Identifying Initial Values
To combat the paradox of the initial run of the EM algorithm we need a separate way to essentially guess the first values to provide, there are a few ways to do this but the 2 that i considered were:

#### Generate Random Starting Numbers
So encompass the entire algorithm in a loop - lets say of 50 iterations - then for each iteration randomly generate new initial starting values, next run the algorithm with the initial values and save the generate and state parameters along with the log-likelihood of the generated states, then proceed onto the next iteration of the model with new starting values.

Once all the iterations have run you then check to see which set of states produced the largest log-likelihood value and use that set of state parameters for the model

#### Clustering Algorithm
The other 'smarter' option I came across was to use a clustering algorithm to roughly find the mean and standard deviation of n clusters - where n is the number of states being generated - then from this we can just retrieve the mean and standard deviation of each cluster and use these as the starting values.

This method drastically improves the run time of the algorithm because instead of having to run the program dozens of times to try and find optimal starting values we spend a bit more computation and time running a clustering algorithm however then it never needs to be ran again so in the long run the runtime is decreased.

Ultimately I chose to use the **Clustering Algorithm** because it provides a more educated approach compared to the random numbers generation's shotgun approach of just throwing numbers at the algorithm and hoping something works well. You can see the code for the clustering algorithm below:

```python
def getInitialValues(data,n_clusters=2,max_iterations=100):
	data_min = min(data)	
	data_max = max(data)
	
	node_positions = np.random.uniform(low=data_min,
										high=data_max,
										size=(n_clusters,))
	
	min_cluster_position_change = 0.01
	
	for i in range(max_iterations):
		clusters = [[] for _ in range(len(node_positions))]
	
		for point in data:
			distances = [abs(point-node) for node in node_positions]
			closest_idx = np.argmin(distances)
	
		clusters[closest_idx].append(point)
	
	new_positions = []
	
	for cluster in clusters:
		if len(cluster)>0:
			new_positions.append(sum(cluster)/len(cluster))
		else:
			new_positions.append(np.random.uniform(low=data_min,high=data_max))
	
	new_positions = np.array(new_positions)

	if max(abs(node_positions-new_positions))<min_cluster_position_change:	
		break
		
	node_positions = new_positions
	final_clusters = [[] for _ in range(len(node_positions))]
	
	for point in data:
		distances = [abs(point-node) for node in node_positions]
		closest_idx = np.argmin(distances)
		final_clusters[closest_idx].append(point)
	
	data_output = []
	
	for cluster,member_nodes in zip(node_positions,final_clusters):
		data_output.append((float(cluster),float(np.std(member_nodes))))
	
	  
	
	return data_output
```

## Expectation Maximisation Algorithm
#### The Likelihood Calculation
To begin with we calculate the likelihood of the target data column - in my code this is the `height` value - for each of the given states - using the initial parameter values from above. This is done by taking each value in each row and using the **Probability Density Function (PDF)** for the gaussian distribution, this is because we are categorising a continuous data value - height - and hence need a continuous distribution, whereas if we were identifying the states for something like a sequence of coin tosses this is a binomial distribution and hence we would use the probability mass function (binomial formula). Once this operation has been performed for each height's likelihood to be in each state these values are then returned.
*The function to get the likelihoods can be seen below*:

```python
def calculateLikelihood(x,state_data):

	mu,sigma = state_data
	y = (1/(sigma*np.sqrt(2*np.pi)))*(np.e**(-0.5*((x-mu)/sigma)**2))
	#probability density function to get likelihood for continuous values
	return y
	
	  

def getLikelihoods(initial_node_data,data):
	for i,state_data in enumerate(initial_node_data):
		#calculate the likelihood for every data point for this state
		data[f'likelihood_{i}'] = data.apply(lambda row:
			calculateLikelihood(row['heights'],state_data),axis=1)
	
	return data
```

#### Probability Calculation / Likelihood Normalisation
The next step of the algorithm is to calculate the probability of each target variable belonging to each of the given states from the likelihoods calculated. This is achieved by just summing the all the likelihood values for a given state and then diving the given likelihood value by the sum of the column - it's essentially just performing normalisation to scale the likelihoods to sum to 1 (hence turning them into the probabilities). 
*The code:*
```python
data = getLikelihoods(state_node_data,original_data)
#get the probabilities for each state
likelihood_cols = [col for col in data.columns if 'likelihood' in col]
sum_series = data[likelihood_cols].sum(axis=1)

for i,target_col in enumerate(likelihood_cols):
	data[f'prob_{i}'] = data[target_col]/sum_series

prob_cols = [col for col in data.columns if 'prob' in col]
```

#### New Mean & Standard Deviation
Now we have the probability that each target value/variable belongs to each state we can calculate the new mean and standard deviation of each state. This is achieved by taking the weighted average of the target variable and the column of probabilities for each state - hence giving us the new mean value for each state. Next from this weighted average we can calculate the variance - which is just volatility squared - by taking each target variable value and subtracting the weighted average (new mean) value to find the difference between the 2 values, then square the difference and multiply it the probability the current target value is in the given state - essentially the weighted average of the square difference between each target variable value and the weighted mean for the current state. Finally we take the square root of this value - as volatility (sigma) is the square root of variance (sigma^2). Finally once this is done for all n identified states the program checks that the maximum change in mean is greater than the threshold that defines the model as converged. If the difference is greater than the threshold value we know the model is still learning otherwise we keep these new mean and volatility values as they represent the mean and volatility for each of the n unobserved states in the data.
*Here's the code for this section of the algorithm:*

```python
new_means = []
new_stds = []

for col in prob_cols:
	weighted_mean = np.average(data['heights'],weights=data[col])
	variance = np.average((data['heights']-weighted_mean)**2,weights=data[col])
	weighted_std = np.sqrt(variance)
	#this mean becomes the new mean & std for that state
	new_means.append(weighted_mean)
	new_stds.append(weighted_std)

  

original_means = np.array([node[0] for node in state_node_data])
new_means = np.array(new_means)

if max(abs(original_means - new_means)) < min_change_threshold:
	print("EM algorithm converged.")
	state_node_data = list(zip(new_means, new_stds))
	break

# This part for updating state_node_data is also fine
state_node_data = list(zip(new_means, new_stds))
```

So all-in-all this update is just introducing the code for the **Markov Switching** Algorithm although it isn't actually applicable for time series forecasting yet as it assumes all data is independent however in reality today's market returns are somewhat linked to previous days' returns as the market carries momentum.