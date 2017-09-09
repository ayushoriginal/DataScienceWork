# Applying Policy Gradient technique to Stock Market (Reinforcement Leaning to use multiple slot machines)

## Overview

I extrapolated the idea of slot machines to the stock market. I built several different trading 'bandit bots' which each have a strategy that they follow. Since the 'brains' of the bots are irregular (compared to a tensor of data for example), they do not lend themselves easily to neural networks. Reinforcement learning is great for this, because it does not require a loss function, only some reward criteria. We can test each of the bots and using the multi-armed bandit algorithm, determine which is the best-performing. 

## Dependencies

* tensorflow (https://www.tensorflow.org/install/)
* numpy
* pandas - for loading in stock data
* matplotlib - for graphhing

## Usage

Install and run `jupyter notebook` 

## Credits

Some base code is taken from [awjuliani](https://github.com/awjuliani).
