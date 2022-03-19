For the # Ensemble method, I started off by finding optimal value of key hyper-parameter, and then built a Random Forest model, and found that for Random Forest, optimal value is 0.838122, when estimator is 400. For comparison, I also built AdaBoost, Gradient Boost and Extra Gradient Boost(XGB) with estimator [50,100,150,200,250,300,350,400,450,500]

Operating on Google Collab, these are the packages I used:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
!pip install category_encoders
import category_encoders
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from category_encoders import OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
