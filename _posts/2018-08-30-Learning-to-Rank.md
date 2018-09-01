---
layout:     post
title:      "Learning to Rank - Overview"
tags:
    - machine learning
    - python
---

Learning to Rank (LTR) is essentially applying supervised machine learning to ranking problems. For ranking problem, we are usually given a set of items and our objective is to find the optimal ordering of this set. For example, in web search, we need to find the best ordering of websites such that websites most relevant to the given query are ordered in the front. 

For the purpose of this note, I will give a quick overview of the LTR framework and provide a demo on how to actually solve a LTR problem in Python with LambdaMART, which is one of the most successful LTR methods.

## LTR Workflow

Similar to other supervised learning, we need a training set. In tasks like classification or regression, a typical training observation has the form of $(x_i, y_i)$ pair with $x_i$ denotes its feature and $y_i$ denotes the corresponding value we are trying to predict. However, LTR is different in the set-up because usually we need to order the results based on a particular input. Using the example of web search again, the optimal ordering of search results depends on what the user is actually looking for. With different queries, the ordering of websites should be quite different. As a result, training set of LTR tasks has the following form:

$$
(q_i, D_i) \\ D_i=\{(d_{i1}, y_{i1}), (d_{i2}, y_{i2}), ...(d_{im}, y_{im})\}
$$

where $q_i$ denotes the query, $D_i$ is an ordered set containing documents $d_{ij}$ together with their relevance score $y_{ij}$ given query $q_i$. 

Once training set is ready, we can train a specific algorithm to learn a ranking model so that for an unseen query and its associated documents so that we can predict the ranking of the documents based on the metric (i.e. loss function) defined when we are training the ranking model. 

## Example with Python

Here is a quick example of how to use LTR in Python using ```xgboost```. As an alternaive, you can also use ```lightgbm``` which has similar interface. 

First, we need to download the dataset from [here](https://s3-us-west-2.amazonaws.com/xgboost-examples/MQ2008.rar). Once download finishes, extract the files and copy `train.txt`, `test.txt` and `vali.txt` from `Fold1` to a desired folder. 

Next, download the data processing script called ```trans_data.py``` from [here](https://github.com/dmlc/xgboost/blob/master/demo/rank/trans_data.py). Make sure to place it in the same folder as `train.txt`, `test.txt` and `vali.txt`. Now open terminal and run the following command:

```bash
python trans_data.py train.txt mq2008.train mq2008.train.group

python trans_data.py test.txt mq2008.test mq2008.test.group

python trans_data.py vali.txt mq2008.vali mq2008.vali.group
```

This will get us the features as well as group information required to perform LTR tasks. 

Now in Python, we first load all the required packages and data. 

```Python
import xgboost as xgb
from xgboost import DMatrix
from sklearn.datasets import load_svmlight_file

# load feature
x_train, y_train = load_svmlight_file("mq2008.train")
x_valid, y_valid = load_svmlight_file("mq2008.vali")
x_test, y_test = load_svmlight_file("mq2008.test")

# load group information
group_train = []
with open("mq2008.train.group", "r") as f:
    data = f.readlines()
    for line in data:
        group_train.append(int(line.split("\n")[0]))

group_valid = []
with open("mq2008.vali.group", "r") as f:
    data = f.readlines()
    for line in data:
        group_valid.append(int(line.split("\n")[0]))

group_test = []
with open("mq2008.test.group", "r") as f:
    data = f.readlines()
    for line in data:
group_test.append(int(line.split("\n")[0]))
```

For ```xgboost```, we need to transform data into a built-in data structure called `DMatrix` before training the model.

```Python
train_dmatrix = DMatrix(x_train, y_train)
valid_dmatrix = DMatrix(x_valid, y_valid)
test_dmatrix = DMatrix(x_test)

# set group information
train_dmatrix.set_group(group_train)
valid_dmatrix.set_group(group_valid)
```

Now training and get prediction from the model will be fairly straightforward. It is similar to classification or regression, except that LTR need an extra group information (which we have incorporated into DMatrix in the previous step).

```Python
params = {'objective': 'rank:pairwise', 'eta': 0.1, 'gamma': 1.0,
               'min_child_weight': 0.1, 'max_depth': 6}
xgb_model = xgb.train(params, train_dmatrix, num_boost_round=4,
                      evals=[(valid_dmatrix, 'validation')])
pred = xgb_model.predict(test_dmatrix)
```

Note that the predicted results give us a score for ordering the test set. Thus before applying any evaluation metric, we need to sort the test set based on the scores first. 