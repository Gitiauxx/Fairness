import pandas as pd


data = pd.read_csv('https://raw.githubusercontent.com/jjgold012/lab-project-fairness/master/fairness_project/datasets/compas-dataset/compas-scores-raw.csv' )
data.to_csv('..\\data\\compas-scores-two-years-raw.csv')
