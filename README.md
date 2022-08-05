# NLP-10-K-10-Q-Alpha-Research
This is an Alpha Research project based on the paper Lazy Prices by Cohen, Malloy and Nguyen (2018). The project aims to construct a high Sharpe long-short portfolio from the S&P500 constituents using only signals based on 10-K/Q (annual/quarterly reports) filings. By extracting singals from textual SEC data using NLP techniques, followed by a portfolio optimization, the resulting model provided the best Sharpe of 1.5 annually. It was also discovered that the strongest signals were coming from the 10-Q’s distance-based features.

# How to read Research Notebooks
### 1. Download Returns
* First part covering S&P500 universe definition, and methods to trace historical changes in constituent stocks to build a Point-in-Time universe.
* Second part is to download all stocks' historical prices (including delisted stocks), using free APIs including Quandl, Yahoo Finance, Alpha Vantage.

### 2. Signal Extraction
* We first web-scrap the SEC Edgar site to obtain its Master Index, which is the full electronic filing list of SEC.
* Next is to perform robust text cleaning to remove noise and identify target sections/paragraphs.
* Apply signal extraction functions to each cleaned document:
  * Distance-based features: Cosine Similarity, Jaccard Similarity, Levenshtein Ratio between count-vectors
  * Embedding-based features: Word2Vec embeddings, Sentence-Encoder followed by a distance function
  * Sentiment-based features: FinBERT, Loughran and McDonald’s Master Dictionary

### 3. Top2Vec for Company Sector Identification
* Topic Modelling approach was used to extract word vectors and document vectors from 10-K's introduction section.
* Clustering (DBSCAN) of document vector produced the topic vectors, which is serving as stock's Sector.

### 4. Machine Learning for Signal Ensembling
* An attempt to use ML algorithm (Learning-to-Rank) to combine various signals into single powerful predictor
* However in this project the LTR method did not show significant improvement compared to non-ML approach

### 5. Signal Analysis
* Produce performance metrics and graphs for each candidate signal
* E.g. Correlation with return, PnL curve, Sharpe, Leverage, Varying investment horizon

### 6. Portfolio Optimization
* Markowitz Portfolio Optimization was done by building a simple covariance model, followed by a convex optimization in CVXPY to produce the final weights.
* Sector Neutrality was set as an option. If true, an extra constrain would be added to CVXPY to force sector total weight to be zero.