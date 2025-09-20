This program has been in the works since February 2025 when I began developing my first simple trading algorithm that just used basic pre-defined indicators combined with a relatively shallow ANN to try and predict the next days's close price for a given stock. This attempt was in-conclusive and couldn't produce any meaningful results.

### The Simpler Approach
After my initial neural network failed I decided to take a step back and try to hard code the trend identifiers. This process meant developing a feature engineering algorithm then trying to develop a strategy from there. I ended up on settling using a momentum based system combined with indicators such as Bollinger Bands to first of all identify if a given stock had momentum and then which direction that momentum was taking the price. However similarly to the first approach this method didn't provide any useful results  

### Back to the Neural Networks
Next up I tried to give the Neural Networks another chance at providing meaningful results, however this time i still used to the feature engineering algorithm to generate all the different technical indicators from the raw stock data. Although instead of just plugging all these into the model I used RFE (Recursive Feature Elimination) to identify the top n number of indicators. To do this I built a simpler XGBoost model to train for each of the hundreds of stocks then used sci kit learn's RFE method to get values for how much each indicator was affecting the output of the model, sorted the features from most to least impactful and finally just select the first 'n' of those results to get the most influential ones on the model prediction algorithm. Furthermore I ended up using the python multiprocessing module which drastically reduced the time it took to run RFE across the hundreds of stocks that were being analysed.

### LSTMs for Context
LSTMs (Long Short Term Memory) work by taking in input data in the format \[samples, Timesteps, Features]. This provides another layer of context compared to the standard neural network, as a normal neural network will just take a single day's data as the input it doesn't contextualise the data the model is seeing. This is where the LSTM reign supreme as by having a different data input format it means that it can see the previous n days' data which can help contextualise the current day's data and hence provide more accurate outputs for the model. For this reason LSTMs become very useful for time series forecasting however due to the computational complexity and the level of parameter tweaking needed to get meaningful results out of the model they're not so widely used as better alternatives have surfaced.

### Current Approach - Markov Chains and Regime switching
I've now switched to using markov chains and regimes, however this method is still in development and I need to do more research into the code behind how to produce this sort of a program.

### Project Updates

| Update Name                                       | Date of Update |
| ------------------------------------------------- | -------------- |
| [[Starting to Implement Markov Chains & Regimes]] | 17/9/2025      |
| [[Improved Back Testing Analytics]]               | 20/9/2025      |




