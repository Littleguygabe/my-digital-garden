# Project Intro

[**Github Link**](https://github.com/Littleguygabe/algo-trading)

This is my first *(somewhat)* working trading algorithm, at a high level it uses principal Component Analysis to identify, take and then hedge positions on assets - from a pre-defined basket - that it believes are mispriced according to statistics and past relationships with other stocks in the basket. However this diverges from being just statistical arbitrage in the way we decide the actual parameters to use because typically we have some parameters to hard code into the system:

1. Window Size for PCA ($m$)
2. Number of principal components to use ($k$)
3. Z-Score threshold for entering a position ($z_{entry}$)
4. Z-Score threshold for exiting a position ($z_{exit}$)

However in this project I build a hyper-parameter tuning engine to optimise these variables using the Sharpe Ratio of the algorithm as the cost function, which will be covered further into the explanation.

## 1. Raw Data

The first step - like in most projects of this nature - is first about getting/defining our data, this is ultimately going to be what we use to first of all identify mispriced stocks and then be the stocks that we use to develop a synthetic portfolio for hedging. This is where we see our first assumption:

**The stocks in our universe (basket) must posses some level of correlation**

So based on this assumption I chose to use a hardcoded basket of tech tickers, which meant they were going to be inherently correlated, however if you're just picking assets there's a couple of easy ways to identify if some stocks are correlated:

### Identify Correlation in stocks

#### Pearson Correlation Coefficient

This method measures the strength of a linear relationship between 2 variables, in our case this is the **returns** - not price - of 2 assets over a fixed period of time. This then gives us a coefficient $c$ where $-1 \le c \le 1$. From this value $c$ we can identify 2 characteristics:

1. The direction of the relationship:
    - $c \lt 0$ - then we know the 2 assets are negatively correlated, so as asset 1's returns decrease, asset 2's returns increase and vice versa
    - $c \gt 0$ - then we know the 2 assets are positively correlated, so as asset 1's returns increase, asset 2's returns increase and vice versa
    - $c = 0$ - then we know the 2 assets have no correlation

2. The strength of the relationship:
    - $0.7 \le |c| \le 1.0$ - Then there is a strong correlation between the 2 assets
    - $0.3 \le |c| \lt 0.7$ - Then there is a moderate correlation between the 2 assets
    - $0.0 \le|c| \lt 0.3$ - Then there is little to no correlation between the 2 assets

Applying this to a full basket involves calculating every pairwise correlation and averaging the result. Typically, for stat arb on a sector-specific basket, we are looking for strong average correlations, ideally $c \ge 0.7$. For classic mean-reversion pairs trading, we need even tighter relationships ($c \ge 0.85$ is common).

#### Scatter Graph

This method is a bit more visual and it just involves taking the returns of each asset in our basket over a given time period and plotting them on a scatter graph, if we find that all the points follow a rough positive or negative trend then we can say they're correlated and can hence perform statistical arbitrage.

#### Dynamic (Rolling) Methods

In the real market, correlations are never static. A relationship might be perfectly stable for years, only for a regime shift (like a market crash, recession, or war) to decouple the stocks.

Using a static coefficient calculated over five years of data is risky. During extreme events like the 2020 crash, many disparate assets gained artificial correlation simply because "everything sold off." To ensure our model trades on active, recent relationships, the algorithm must use a dynamic rolling window for calculating correlation.

### My Data

So with my defined basket:

``` python
basket = [
    'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'BRK-B', 'WMT',
    'LLY', 'JPM', 'V', 'ORCL', 'XOM', 'JNJ', 'MA', 'NFLX', 'PLTR', 'ABBV',
    'COST', 'BAC', 'AMD', 'HD', 'PG', 'GE', 'CSCO', 'CVX', 'KO', 'UNH',
    'IBM', 'WFC', 'CAT', 'MS', 'AXP', 'MU', 'GS', 'MRK', 'CRM', 'TMUS',
    'PM', 'APP', 'RTX', 'MCD', 'ABT', 'TMO', 'AMAT', 'ISRG', 'PEP', 'LRCX'
    ]
```

I built a GitHub workflow that at the end of the week when the US market closes (~10pm GMT) the workflow automatically removes all the old data stored in the `data/historical_data` folder and updates it with the most recent 5 years of historical data, this means that I don't have to constantly make requests to YFinance for ticker data, and that I have access to the data even if I don't have an internet connection (it also updates data for a set of random walks, however this doesn't work because it breaks the assumption that the assets in the basket are correlated). The script for updating the historical data is shown below:

```python
import pandas as pd
import yfinance as yf
import logging
import os
logging.basicConfig(level=logging.INFO)

basket = [
    'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'BRK-B', 'WMT',
    'LLY', 'JPM', 'V', 'ORCL', 'XOM', 'JNJ', 'MA', 'NFLX', 'PLTR', 'ABBV',
    'COST', 'BAC', 'AMD', 'HD', 'PG', 'GE', 'CSCO', 'CVX', 'KO', 'UNH',
    'IBM', 'WFC', 'CAT', 'MS', 'AXP', 'MU', 'GS', 'MRK', 'CRM', 'TMUS',
    'PM', 'APP', 'RTX', 'MCD', 'ABT', 'TMO', 'AMAT', 'ISRG', 'PEP', 'LRCX'
    ]

output_dir = 'data/historical_data'

logging.info("Updating Historical Data")

for filename in os.listdir(output_dir):
    os.remove(os.path.join(output_dir,filename))

for ticker in basket:
    yf_ticker = yf.Ticker(ticker)
    data = yf_ticker.history(period='5y')
    data.to_csv(os.path.join(output_dir,f'{ticker}.csv'))
    logging.info(f"{ticker} data updated")

logging.info("Finished Updating Historical Data")
```

## 2. Data Processing

We begin by retrieving the adjusted close price for each stock and converting this into a matrix of day-on-day returns which is then centred and standardised to have a mean of 0 and standard deviation of 1. This matrix $Z$ is defined as follows:

$$Z \in \mathbb{R}^{m \times n}$$

- $m$: Number of days (time rows)
- $n$: Number of stocks (asset columns)

We then perform PCA on Z to obtain the eigenvalues and eigenvectors (eigen-portfolios).

This is the point in the program where we need to decide how many principal components ($k$) that we want to use, which presents a relatively complex optimisation problem because say we use $n$ principal components then yes this would explain all of the variance in price however it also picks up on idiosyncratic noise (small per-stock factors) so we won’t be able to find a large enough spread on the residuals to trade. On the flip side though if we only use 1 principal component then this will likely result in us just mapping the market beta and as a result we won’t be able to find any opportunities for an arbitrage strategy.

So to find the solution to this optimisation problem I simply gave the hyper-parameter tuner a range of values for both the number of principal components $k$ and the window size $m$ and chose the pairing that gave the best sharpe ratio where:

- $3 \le k \le 13$
- $300 \le m \le 501$

I found that taking $k = 7$ provided the best amount of variance explained but still gave the ability to arbitrage, however choosing the best $m$ was slightly more complex as it also varied with with $z_{entry}$, so I arrived to taking $m = 392$, and the rationale will be covered when I come to talking about the $z$ threshold values later on. However as a rough baseline on an MFT like this one you use the bounds

$$3 \cdot n_{assets} \le m \le 5 \cdot n_{assets}$$

- $n_{assets}$ is the number of assets in our tradeable universe.
- $m$ is the number of days of data being used (window size)

### Alternative Methods for Choosing $k$

#### 1. Scree Plot (Elbow Method)

We plot the each principal component ($k$) on the X-axis against the individual amount of variance (eigenvalue, $\lambda$) that is explains on the Y-axis. Then we look to see where the curve generated starts to taper out (ie finding the 'Elbow') which is the point where adding another principal component only explains a minimal amount of variance. This means that any components from this point are are just explaining idiosyncratic noise in the financial data rather than market factors, so we cut off $k$ at that point.

#### 2. Cumulative Variance Threshold

This method first asks how much variance of the basket of stocks returns matrix ($Z$) do we want to explain. The issue with this approach is the risk of overfitting to the financial data, because in statistical arbitrage any profit is generated from the residuals ($\epsilon$):
$$\text{Stock} = \text{Explained by PCA} + \text{Residual}(\epsilon)$$

- So if we have 99% of variance explained by PCA then our residual is only 1%, which is too small of a spread to be able to properly trade
- Whereas if 50% of variance is explained by PCA then our residual is going to sit at 50%, which is a big enough spread to trade on.

The other thing to note though is that we don't want to explain too little variance with PCA because otherwise we begin to introduce higher levels of risk into our strategy, so ideally we want to find that sweet spot of being structured for safety but having enough noise to be profitable.This approach is relatively easy to implement into code though because we just follow the formula:
$$\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{n} \lambda_j} \geq \text{Threshold}$$

- $k$: the number of principal components
- $n$: total number of stocks in our basket
- $\lambda$: the eigenvalues (principal components) generated from PCA

#### 3. Marchenko-Pastur Distribution Theory (Random Matrix Theory)

This is the most complex out of the 3 methods listed but in reality its a fairly simple concept, essentially if an eigenvalue ($\lambda$) is below some upper threshold ($\lambda_+$) then we know from the Marchenko-Pastur Distribution Theory that a similar value for $\lambda$ could also be generated from PCA performed on random noise, so we discard it as it doesn't represent a significant market factor. This upper threshold for eigenvalues ($\lambda_+$) can be calculate with the following formula:
$$\lambda_+ = \sigma^2 \left( 1 + \sqrt{\frac{n}{m}} \right)^2$$

- $\sigma^2$: the variance of the residuals, however as we standardised the data this is just **1**
- $n$ is the number of stocks in our basket
- $m$ is the number of days of data we're using

Then once we have our upper bound for the eigenvalues ($\lambda_+$)  any eigenvalue whose weight is less than $\lambda_+$ is then discarded as it likely represents idiosyncratic noise not a significant market factor.

## 3. Calculating Factor Returns

So now we've performed PCA using $k$ components on a data window of $m$ to get our selected eigenvectors (let's call this $V_k$) where each value in each row represents how much a given stock contributes to a given component, this is a bit easier when explained slightly more mathematically:

The vector of eigenvalues $\Lambda$ tells you the "power" of each factor. If we have $n$ stocks, the total variance of the system is the sum of all eigenvalues. The proportion of total variance explained by the $i$-th component is:

$$\text{Variance Explained Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{n} \lambda_j}$$

Where: $\lambda_i$ is the $i$-th eigenvalue. The eigenvalues are typically sorted such that $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_k$.

The Loading Matrix (Eigenvectors) $V_k$ contains the "recipes" for your $k$ factors. Each row is an eigenvector representing a Principal Component.

$$V_k = \begin{bmatrix} v_{1,1} & v_{1,2} & \dots & v_{1,n} \\v_{2,1} & v_{2,2} & \dots & v_{2,n} \\\vdots & \vdots & \ddots & \vdots \\v_{k,1} & v_{k,2} & \dots & v_{k,n}\end{bmatrix}$$

For a specific Principal Component $i$, the vector is:$$\mathbf{v}_i = [v_{i,1}, v_{i,2}, \dots, v_{i,n}]^T$$

Each value $v_{j,i}$ is the weight of stock $j$ used to construct the $i$-th factor. Hence the returns of the $i$-th Principal Component at time $t$ ($P_{i,t}$) are calculated as the weighted sum of the individual stock returns ($R_{j,t}$):

$$P_{i,t} = \sum_{j=1}^{n} v_{j,i} R_{j,t}$$

So now to get the factor returns we simply take our complete window of standardised returns and project the transposition of our eigenvectors onto them:

$$\underset{(m \cdot n)}{\text{Stock Returns}} \cdot \underset{(n \cdot k)}{\text{Eigenvectors}^T} = \underset{(m \cdot k)}{\text{Factor Returns}}$$

Mathematically:

$$Z \cdot V_k^T = F$$

After projecting our standardised stock returns $Z$ onto the principal components $V_k$, we obtain the Factor Return Matrix $F \in \mathbb{R}^{m \times k}$. Each element $f_{i,j}$ represents the specific return of Factor $j$ on Day $i$ within our lookback window:

$$F = \begin{bmatrix} f_{1,1} & f_{1,2} & \dots & f_{1,k} \\f_{2,1} & f_{2,2} & \dots & f_{2,k} \\\vdots & \vdots & \ddots & \vdots \\f_{m,1} & f_{m,2} & \dots & f_{m,k} \end{bmatrix}$$

Where:

- $i \in \{1, \dots, m\}$ denotes the trading day in the window.
- $j \in \{1, \dots, k\}$ denotes the principal component (factor).

## 4. Statistical Arbitrage Regression

So now we have all the components we need to actually start building trading signals, we just need to do a couple of multiplcation operations, find some coefficients and define some thresholds and we can start trading (in a very simplistic way)

### Beta Weights

So lets say we want to see if there is an opening to perform arbitrage on the $i^{th}$ stock in our universe. We need to find out how much of each factor's returns contribute to the return of the $i^{th}$ stock which is done through linear regression where we take our target stock's return as the target value and the factor weights on a given day as the input values. so mathematically:

For stock $i$, we extract its column of standardised returns from $Z$:

$$\mathbf{z}_i = \begin{bmatrix} z_{1,i} \\ z_{2,i} \\ \vdots \\ z_{m,i} \end{bmatrix} \in \mathbb{R}^m$$

We then model this stock's returns as a linear combination of the $k$ factor returns, plus some residual $\boldsymbol{\epsilon}_i$:

$$\mathbf{z}_i = F\boldsymbol{\beta}_i + \boldsymbol{\epsilon}_i$$

Where $\boldsymbol{\beta}_i \in \mathbb{R}^k$ is the vector of beta weights we want to estimate:

$$\boldsymbol{\beta}_i = \begin{bmatrix} \beta_{i,1} \\ \beta_{i,2} \\ \vdots \\ \beta_{i,k} \end{bmatrix}$$

Each $\beta_{i,j}$ tells us how sensitively stock $i$ responds to factor $j$. We find the $\boldsymbol{\beta}_i$ that minimises the sum of squared residuals, which has the closed-form Ordinary Least Squares (OLS) solution:

$$\hat{\boldsymbol{\beta}}_i = (F^T F)^{-1} F^T \mathbf{z}_i$$

This is computed for every stock $i \in \{1, \dots, n\}$, giving us the full Beta Matrix:

$$B = \begin{bmatrix} \hat{\boldsymbol{\beta}}_1 & \hat{\boldsymbol{\beta}}_2 & \cdots & \hat{\boldsymbol{\beta}}_n \end{bmatrix} \in \mathbb{R}^{k \times n}$$

The residual vector for stock $i$ — the part of its returns *unexplained* by the $k$ factors — is then:

$$\hat{\boldsymbol{\epsilon}}_i = \mathbf{z}_i - F\hat{\boldsymbol{\beta}}_i$$

It is precisely this residual $\hat{\boldsymbol{\epsilon}}_i$ that we monitor for trading signals. If the factors fully explained stock $i$'s returns, $\hat{\boldsymbol{\epsilon}}_i \approx \mathbf{0}$ and there would be nothing to trade. Any persistent deviation from zero implies the stock is mispriced relative to its factor exposures — the opportunity our strategy seeks to exploit.

## 5. Z-Score Thresholding

Ok, so we now have the residual $\hat{\boldsymbol{\epsilon}}$ we need to establish how we actually convert this into a trading signal for either entering or exiting a position.

For finding a entry we are trying to identify a stock that is unusually far from where we would expect it to be, the keyword here being unusually, because we always expect some level of idiosyncratic noise. So we just need to find a residual so large that we can statistically say it's not just idiosyncratic noise.

Then once the stock's residual reduces back to a normal level that's exactly when we want to exit our position as the price has now reverted back to the mean level of idiosyncratic noise - why we need the mean-reversion assumption. So we just need to define what is a 'normal' level of idiosyncratic noise.

So as a general baseline we take:
$$z_{entry} = 2 \\z_{exit} = 0.25$$

However these values later change as we use hyper-parameter tuning to optimise the parameters using the sharpe ratio as our cost function.

To calculate the Z-score of the current data, we use the return residuals over a specific rolling window. Let $\epsilon_t$ be the return residual at time $t$, and let $W$ represent the size of our rolling window. First, calculate the rolling mean ($\mu_t$) of the residuals:$$\mu_t = \frac{1}{W} \sum_{i=0}^{W-1} \epsilon_{t-i}$$

Next, calculate the rolling standard deviation ($\sigma_t$) of the residuals:$$\sigma_t = \sqrt{\frac{1}{W} \sum_{i=0}^{W-1} (\epsilon_{t-i} - \mu_t)^2}$$

Next, calculate the current Z-score ($Z_t$), which determines exactly how many standard deviations the current residual is from the rolling average:$$Z_t = \frac{\epsilon_t - \mu_t}{\sigma_t}$$

Finally we can now decide if we actually want to put a position on based on the current asset we're looking at:

- $|Z_t| \geq z_{\text{entry}}$ - This gives us our signal to enter a position in one of 2 directions:
    1. $Z_t \lt 0$ -  The stock is underpriced, so we go long on the stock expecting it's value to go **up** towards the mean.
    1. $Z_t \gt 0$ - The stock is overpriced, so we short the stock expecting it's value to return down towards the mean.

- $|Z_t| \leq z_{\text{exit}}$ - This gives us our signal to unwind a position as the residual has moved back to it's usual values.

## 6. Trading our Signals

Now we have everything we need to actually be able to start trading, how do we actually trade the strategy?

### 1. Position Entry

As mentioned above this is just dictated by Z-score of the residual of one of our assets on a given day, so once we have a $|Z_t| \gt z_{entry}$ we make the call to either long or short that given asset.

### 2. Setting up the Position

If we just either went short or long on our stock this isn't true arbitrage, it's just a direction bet that some asset is going to eventually mean revert, so to make this closer to true arbitrage (this strategy will never be true arbitrage as fat-tail events will always exist so we'll never to be truly risk free) we need to find a way to hedge our position.

This is done by creating what we call a **replicating portfolio** which is essentially making a combination of all the other stocks in our tradeable universe that replicates the returns of the stock we want to trade. Then that means we can just take an opposite position on this portfolio, ie our target stock's returns go up, the replicating portfolio goes down so we stay neutral, which allows us to just trade the noise.

So how do we make this replicating portfolio?

### 3. Building the Replicating Portfolio

#### Finding the Hedge Ratios

Conveniently this can be done quite naturally using the same matrices that we already calculated for the initial residual calculation. If you remember we have a couple of main matrices which can be used again:

- $V_k$ - This is our eigenvectors from [step 3](#3-calculating-factor-returns) representing how much each stock in our basket contributes to a given factor's returns
- $B$ - These are the beta weights we calcualted during [step 4](#4-statistical-arbitrage-regression) which show how much each factor contributes to each stock's returns.

To determine the exact quantity of each stock to trade as a hedge, we calculate the dot product of our two input matrices ($V_k$ and $B$). In the resulting matrix, each row represents the specific replicating portfolio required to hedge the stock associated with that row.

Mathematically, this is expressed as:
$$V_h = V_k \cdot B$$

Where:

- $V_h \in \mathbb{R}^{n \times n}$ is the resulting matrix of hedge ratios.
- $V_k$ and $B$ are the component matrices being multiplied.
- Each row $i$ in $V_h$ dictates the weights of the assets needed to form the replicating portfolio for stock $i$.


Now as you might've noticed, $V_k$ is a $ k \cdot n$ matrix showing how we go from asset returns to factor returns and $B$ is a $n \cdot k$ matrix showing how we go from factor returns to asset returns. So instead of having to re-calculate the beta coefficients and perform linear regression everytime we want to hedge a trade we can actually take the transposition of our eigenvectors $V_k^T$ as our beta coefficients $B$, hence:

$$ v_{i,j} = u_{j,i} $$

Where:

- $v_{i,j}$ is the value at $(i,j) \in V_k^T$
- $u_{j,i}$ is the value at $(j,i) \in B$

#### Removing the Target Stock From its own Hedge

However this isn't quite the final step, although we have our hedge weights for each stock in the basket in terms of every stock in the basket, it means that these hedge ratios also tell us how much of our target stock we need to trade to hedge against our target stock, so we can have a scenario of saying for every 1 unit bought of stock A, the hedge we sell 1 unit of stock A which obviously doesn't work.

So to combat this we just have to use some scaling, for a given set of hedge ratios (row) in $V_h$ we look at the amount of our target stock we have to hedge against itself $x$ and then we simply divide all other values by $1 - x$ so that they now carry all the weight for the hedge.

For example:
Imagine the replicating portfolio says that 100% of $AAPL$'s price movement can be reconstructed like this:

- 20% is explained by $AAPL$ itself
- 80% is explained by other assets in our universe

If we want to build a replicating portfolio for $AAPL$ using only other stocks we run into a problem, our mix of other stocks only gives us 80% of the coverage of $AAPL$'s returns. So if we just get rid of $AAPL$ from our replicating portfolio we have a hedge that is only 80% the size it should be - so we aren't really hedged.

So we need to inflate that remaining 80% to cover the full 100%.

Scaling that 0.8 (80%) is fairly trivial, as we just need to answer *'How do we turn 0.8 into 1.0?'*. This is achieved by simply dividing by 0.8, which comes from $1 -$ our target stock's ($AAPL$) hedge weight.

Because our hedge ratio matrix $V_h$ dictates how much of one stock is used to hedge another, the diagonal elements of this matrix, $(V_h)_{i,i}$, represent how much of stock $i$ is used to hedge itself. To mathematically remove this self-weight, we create a scaling vector $S$ where each element is defined as:

$$S_i = 1 - (V_h)_{i,i}$$

We then set the diagonal elements of our matrix to zero ($(V_h)_{i,i} = 0$) and divide each row $i$ of $V_h$ by its corresponding scaling factor $S_i$. This scales our remaining hedge ratios, mathematically expressed as $\frac{(V_h)_{i,j}}{S_i}$ for all $j \neq i$, ensuring the replicating portfolio provides full coverage without including the target stock in its own hedge.

### 4. Finally Trading

Now we truly have everything we need to execute a trade. Let $x$ represent the quantity of the target asset we want to trade, and let $W_i$ be the vector of our scaled hedge ratios for that specific target asset. 

To determine the exact position sizes for our hedging basket, denoted as $P_i$, we multiply our hedge ratio vector by $-x$:

$$P_i = -x \cdot W_i$$

The negative sign ensures we take an opposing position in the replicating portfolio (e.g., shorting the basket) to properly hedge our long position of size $x$ in the target asset. Finally, we execute the trades based on the values in $P_i$.