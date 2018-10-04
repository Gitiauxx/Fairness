import pandas as pd

# compas data
admissions = pd.read_csv('https://raw.githubusercontent.com/jjgold012/lab-project-fairness/master/fairness_project/datasets/ucla-law-school-dataset/lsac.csv')
admissions.to_csv('..\\data\\admissions_bar.csv')