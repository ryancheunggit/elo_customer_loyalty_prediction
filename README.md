# [Elo Merchant Category Recommendation](https://www.kaggle.com/c/elo-merchant-category-recommendation)
10 days hack for 66th place 

As my time was limited, I tried not to try too out of box stuff, no fancy features creation, no fancy feature selection, no post-processing, no massive ensemble. 

1. Shifted transactional datasets.
  + The data in historical transaction all have `month_lag <= 0`, and the data in new transaction all have `month_lag > 1` plus the condition that the combination of `card_id & merchant_id` in new does not exist in historical. 
  + The preprocessing step is to mimic the historical transaction and new transaction split done by elo for each customer. I shifted the dividing `month_lag` forward and generate shifted version of new transactions. 
  + These shifted version of dataset were still useful for predicting customer loyalty scores, obviously not as predictive as the full dataset, but models built on shifted datasets were forced to learn things that's not most recent. 
2. Features.
  + For transactional data, it is just some groupby(categoricals) and aggregation on purchase amounts. 
  + On top of ^, there were some interaction terms(mainly ratios) created.
  + There were also some dates related features extracted on purchase dates alone.(the diffs between dates were very predictive)
3. Models.
  + Layer1 are mostly GBMs trained on features generated on different datasets. 
  + To incorporate those `outliers` (which I suspect to be customers that canceled accounts) better, I added in strong classification models to the mix. 
  + And also throwed in RGF model just for a bit extra diversity. (Wish I could have time to train some NN). 
  + Layer2 is just a simple Ridge regression.
4. Things tried without luck.
  + Rank models. 
    - For LightGBM, to use `lambdarank`, we need to specify the `query/grouping` to the dataset, I tried different grouping methods, but couldn't really make a decision to include it in the mix or not.
    - For XGBoost, to use `rank:pairwise`, we don't need to specify query, but I had hard time to get the training working, it terminates very quickly. 
    - Having little experience working with rank models, I decided to exclude them in the final mix. 
    
Overall, my local 5 folds CV is very consistent in the 3.642 ~ 3.645 zone, even though the associated PLB score is above 3.691, I was quite confident I could shake up during the leaderboard swap. 

