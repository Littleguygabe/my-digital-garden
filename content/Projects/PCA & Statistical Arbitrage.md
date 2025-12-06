[**Github Link**](https://github.com/Littleguygabe/Small-Projects)
*under principle-component-analysis/*

This project is my first real look into quant research methods, by performing principle component analysis on a basket of stocks and then using a mean reverting method combined with Z-scores to produce trading signals. 

However I'm still researching and learning about the Statistical Arbitrage side of this and how I can actually use the beta values from regression between the principle components and my synthetic personal portfolio to actually produce trading signals, so as I figure things out I'll link the updates on this page.


### Rough Workflow

First, we need to define a basket of stocks to develop our eigen-portfolios. Eventually, these will be used to hedge the stock we want to buy. For this first implementation of the project I'll just be using a hardcoded basket of stocks to look at, this is the current 50 largest tickest in the S&P 500 by weight.

```python
basket = [
'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'BRK-B', 'WMT',
'LLY', 'JPM', 'V', 'ORCL', 'XOM', 'JNJ', 'MA', 'NFLX', 'PLTR', 'ABBV',
'COST', 'BAC', 'AMD', 'HD', 'PG', 'GE', 'CSCO', 'CVX', 'KO', 'UNH',
'IBM', 'WFC', 'CAT', 'MS', 'AXP', 'MU', 'GS', 'MRK', 'CRM', 'TMUS',
'PM', 'APP', 'RTX', 'MCD', 'ABT', 'TMO', 'AMAT', 'ISRG', 'PEP', 'LRCX'
]
```

## 1. Data Preparation and PCA

We begin by retrieving the adjusted close price for each stock and converting this into a matrix of day-on-day returns which is then centred and standardised to have a mean of 0 and standard deviation of 1. This matrix $Z$ is defined as follows:

$$Z \in \mathbb{R}^{m \times n}$$

- $m$: Number of days (time rows)
- $n$: Number of stocks (asset columns)

We then perform PCA on $Z$ to obtain the eigenvalues and eigenvectors (eigen-portfolios).

This is the point in the program where we need to decide how many principle components ($k$) that we want to use, which presents a relatively complex optimisation problem because say we use $n$ principle components then yes this would explain all of the variance in price however it also picks up on idiosyncratic noise (small per-stock factors) so we won't be able to find a large enough spread on the residuals to trade. On the flip side though if we only use 1 principle component then this will likely result in us just mapping the market beta and as a result we won't be able to find any opportunities for an arbitrage strategy.

So currently my program uses $k$ = 15 (15 most significant principle components) however I'm looking into methods to optimise this, some examples of methods are:
##### 1. Scree Plot (Elbow Method)
We plot the each principle component ($k$) on the X-axis against the individual amount of variance (eigenvalue, $\lambda$) that is explains on the Y-axis. Then we look to see where the curve generated starts to taper out (ie finding the 'Elbow') which is the point where adding another principle component only explains a minimal amount of variance. This means that any components from this point are are just explaining idiosyncratic noise in the financial data rather than market factors, so we cut off $k$ at that point.
##### 2. Cumulative Variance Threshold
This method first asks how much variance of the basket of stocks returns matrix ($Z$) do we want to explain. The issue with this approach is the risk of overfitting to the financial data, because in statistical arbitrage any profit is generated from the residuals ($\epsilon$):
$$\text{Stock} = \text{Explained by PCA} + \text{Residual}(\epsilon)$$
- So if we have 99% of variance explained by PCA then our residual is only 1%, which is too small of a spread to be able to properly trade
- Whereas if 50% of variance is explained by PCA then our residual is going to sit at 50%, which is a big enough spread to trade on.

The other thing to note though is that we don't want to explain too little variance with PCA because otherwise we begin to introduce higher levels of risk into our strategy, so ideally we want to find that sweet spot of being structured for safety but having enough noise to be profitable.This approach is relatively easy to implement into code though because we just follow the formula:
$$\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{n} \lambda_j} \geq \text{Threshold}$$
- $k$: the number of principle components
- $n$: total number of stocks in our basket
- $\lambda$: the eigenvalues (principle components) generated from PCA
##### 3. Marchenko-Pastur Distribution Theory (Random Matrix Theory)
This is the most complex out of the 3 methods listed but in reality its a fairly simple concept, essentially if an eigenvalue ($\lambda$) is below some upper threshold ($\lambda_+$) then we know from the Marchenko-Pastur Distribution Theory that a similar value for $\lambda$ could also be generated from PCA performed on random noise, so we discard it as it doesn't represent a significant market factor. This upper threshold for eigenvalues ($\lambda_+$) can be calculate with the following formula:
$$\lambda_+ = \sigma^2 \left( 1 + \sqrt{\frac{n}{m}} \right)^2$$
- $\sigma^2$: the variance of the residuals, however as we standardised the data this is just **1**
- $n$ is the number of stocks in our basket
- $m$ is the number of days of data we're using

Then once we have our upper bound for the eigenvalues ($\lambda_+$)  any eigenvalue whose weight is less than $\lambda_+$ is then discarded as it likely represents idiosyncratic noise not a significant market factor.  

### 2. Constructing Eigen-Portfolio Returns

Once we have our selected eigenvectors (let's call this matrix $V_k$), we project our original returns onto these vectors to see how the "hidden" factors performed over time.

We perform a matrix multiplication of our standardised returns ($Z$) against the transposed eigen-portfolios ($V_k$). This looks like this:

$$\underset{(m \times n)}{\text{Stock Returns}} \times \underset{(n \times k)}{\text{Eigenvectors}^T} = \underset{(m \times k)}{\text{Factor Returns}}$$

Mathematically:

$$Z \cdot V_k = F$$

The resulting matrix $F$ has dimensions $m \times k$, where each column represents the daily returns of a specific eigen-portfolio.

### 3. Statistical Arbitrage Regression

Now we perform the regression to find the trading signal. We select a target stock from our basket (let's call its returns $Y_{target}$) and use the eigen-portfolio returns ($F$) as our independent variables.

We solve for the coefficients ($\beta$) in the following structure:

$$\underset{(m \times 1)}{Y_{target}} = \alpha + \left( \underset{(m \times k)}{F} \cdot \underset{(k \times 1)}{\beta} \right) + \epsilon$$

- **$Y_{target}$**: The actual returns of the stock we want to trade.
- **$F \cdot \beta$**: The "synthetic" version of the stock constructed using our principal components.
- $\epsilon$ : The residual, this is the difference between the actual stock and its theoretical PCA value.
- $\alpha$: Any return we've made 'beating' the market

We can then rearrange this equation to find the residual ($\epsilon$):

$$\underset{(m \times 1)}{Y_{target}} - \alpha - \left( \underset{(m \times k)}{F} \cdot \underset{(k \times 1)}{\beta} \right) = \epsilon$$
And from here we look at the residual of $Y_{target}$  and calculate its Z-score to identify whether this is a wide enough spread for us to enter a trade, which then also leads us into how we would hedge this position after it's been identified. But as I'm still working on that, I'll post an update when the code is done for it.