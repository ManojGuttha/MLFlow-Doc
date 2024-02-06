# SETUP

a. After executing Sklearn regression code, There will be a new directory formed called "MLruns"
b. We gonna explore and keep the track of the experiments with the use of MLFlow tour
c. To explore the MLFlow web tool, First we gonna activate the mlflow library in conda environment

Command to activate: "conda activate [projectfilename]"
Command to get the URL for MLFlow: "mlflow ui"

This is how initial user interface of MLFlow looks like


![one](https://github.com/ManojGuttha/MLFlow-Doc/assets/158753894/3d387339-3cb1-4d64-87b5-c137369f55b1)


There are two sections called: Experiments and Models

In MLFlow, Experiments may contain n number of runs. A "run" indicates the number of times the code is executed.
In the below picture as you can see there are 3 runs in ths experiment. We can filter out the runs data as we like.


![image](https://github.com/ManojGuttha/MLFlow-Doc/assets/158753894/85b0d1c6-6888-4522-8c82-0c83c44c5970)


From the above picture, we can track the model's r2 and rmse score for different alpha and l1_ratio.
We can also compare 2 different runs in an experiment to track their performance.


![image](https://github.com/ManojGuttha/MLFlow-Doc/assets/158753894/3b3057aa-f8bf-4e0b-9514-d0a1080d7728)


We can see it in different graphical options


![image](https://github.com/ManojGuttha/MLFlow-Doc/assets/158753894/0127a4e5-86b3-4a46-af23-3c9048e411f4)
![image](https://github.com/ManojGuttha/MLFlow-Doc/assets/158753894/6ecbf08d-c34e-4d25-a65d-aae366dda54a)
![image](https://github.com/ManojGuttha/MLFlow-Doc/assets/158753894/71302c1c-e125-4992-8ec9-e9d327323a98)


Each run stores the data of Datasets, Parameters, Metrics, Tags and Artifacts as shown below


![image](https://github.com/ManojGuttha/MLFlow-Doc/assets/158753894/468ab711-e4a0-4097-82d5-82bcc6093210)

### Code

```python

## Sklearn Regression Model with MLFlow

import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()

# evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from local
    data = pd.read_csv("C:/Users/Manoj/Desktop/mlflowcourse/red-wine-quality.csv")
    data.to_csv("red-wine-quality.csv", index=False)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio
    exp = mlflow.set_experiment(experiment_name="experment_1")

    with mlflow.start_run(experiment_id=exp.experiment_id):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(lr, "mymodel")

```
