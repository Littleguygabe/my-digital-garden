This update will be fairly short but sweet - I've been quite busy lately what with uni starting back up amongst other things so not too much has changed.

When identifying unobserved states to use for financial time series analysis the main purpose of the algorithm is to be able to predict some feature - return in this case - for the next day. So lets say this wasn't being done through markov regimes the standard approach could be to use something like a regression model where you take an auto-regressive order - which is just the number of previous days' data you take into account to predict the next day's feature value - then use this to generate a regression algorithm. For a regime switching model this means fitting a regression model to each different state within the system which presents its own challenges.

One of these challenges is how to make sure that for a given state's regression model how do we make sure that the model is only being fit to the data that is actually in that state, because if we fit the regression model to all data for all states then there's no point having the states. This is achieved by weighting each of values in the data with how likely it is to be in the given state.

### For example:

| Value | P(s1) | P(s2) |
| ----- | ----- | ----- |
| 14    | 0.7   | 0.3   |
| 8     | 0.4   | 0.6   |
| 19    | 0.95  | 0.05  |
| 2     | 0.1   | 0.9   |
So on in the data above we can quite clearly see that values above 10 are in state 1 and below 10 are in state 2, this means when we fit a regression algorithm when we are in state 1 we don't want the algorithm to heavily fit towards the values that are likely to be in state 2, and similarly for state 2, we don't want the values that are likely to be in state 1 to have a strong influence over the regression algorithm because simply they aren't likely to happen in the current market.

This is why we weight the values when fitting the regression for each state, so for state 1 the values we are actually fitting to are `[9.8,3.2,18.05,0.2]` meaning the coefficients we generate for each value in the regression algorithm will be more targeted towards the current state.

Once we have the regression parameters created we can then re-calculate the probabilities using a likelihood function then normalising each values likelihood of being in a given state, this then provides us with new probabilities which can then be fed back into the theta estimation algorithm and this iterative process repeats until the values for theta converge within a given threshold.
