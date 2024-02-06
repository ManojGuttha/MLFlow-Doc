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
