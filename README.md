# Smart-Portfolio-Management-with-Deep-Learning

Welcome to this repository! This project dives into the exciting world where Artificial Intelligence meets finance. We're exploring how deep learning models, specifically LSTMs and Transformers, can be used to build smarter investment portfolios and allocate assets more effectively.

# What's This All About?
At its core, this project aims to predict future asset returns and then use those predictions to optimize a portfolio. Instead of relying purely on traditional methods, we're leveraging the power of neural networks to uncover patterns in financial data that might otherwise be missed.

We've set up a full workflow here:

Data Collection: Pulling historical stock market and foreign exchange data.

Feature Engineering: Transforming raw data into meaningful signals (like returns, moving averages, and volatility).

Deep Learning Models: Training and comparing two advanced neural network architectures – Long Short-Term Memory (LSTM) networks and Transformer models – to forecast asset returns.

Portfolio Optimization: Using the predicted returns, along with historical risk, to determine optimal asset allocations using techniques like Mean-Variance Optimization.

Backtesting: Simulating how our strategies would have performed historically, complete with transaction costs, and comparing them against traditional benchmarks (Equal Weight and Static Mean-Variance).

Performance Analysis & Saving Results: Evaluating our models' financial performance and saving all key data, models, and charts for easy review.

How to Get Started
To run this project on your machine, follow these simple steps:

1. Clone the Repository
First, grab a copy of this code:
```console
$ git clone https://github.com/sumitpardhiya/Smart-Portfolio-Management-with-Deep-Learning.git
$ cd Smart-Portfolio-Management-with-Deep-Learning
```
2. Set Up Your Environment

This project relies on several Python libraries. It's highly recommended to use a virtual environment to keep things tidy.

# Create a virtual environment
```console
$ python -m venv venv
```
# Activate the virtual environment
# On Windows:
```console
.\venv\Scripts\activate
```
# On macOS/Linux:
```console
source venv/bin/activate
```

# Install the required packages
```console
pip install pandas numpy scikit-learn tensorflow yfinance pypfopt matplotlib seaborn
```

3. Run the Code

Once your environment is ready, simply execute the main script:
```console
$ python quantitive_finance.py
```
# The script will:

- Download historical financial data.

- Process and engineer features.

- Train the LSTM and Transformer models.

- Run backtests for each strategy and benchmarks.

- Print performance metrics to your console.

- Generate and display a cumulative portfolio value chart.

# What You'll Find After Running:
Upon successful execution, the following files will be generated in your project directory:

- downloaded_prices_raw.csv: Raw historical adjusted close prices.

- downloaded_volumes_raw.csv: Raw historical trading volumes.

- lstm_model.h5: The trained LSTM model.

- transformer_model.h5: The trained Transformer model.

- lstm_portfolio_values.csv: Daily portfolio values for the LSTM strategy during backtesting.

- transformer_portfolio_values.csv: Daily portfolio values for the Transformer strategy during backtesting.

- ew_portfolio_values.csv: Daily portfolio values for the Equal Weight benchmark.

- static_mvo_portfolio_values.csv: Daily portfolio values for the Static Mean-Variance Optimization benchmark.

- cumulative_portfolio_value.png: A plot visualizing the cumulative performance of all strategies and benchmarks.
