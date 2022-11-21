import os
import modal
from sklearn.preprocessing import LabelEncoder

    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()
    titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")

    #Feature engineering
    titanic_df = titanic_df.drop(columns=["Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"])

    titanic_df["Age"] = titanic_df["Age"].fillna(-1)
    titanic_df["Fare"] = titanic_df["Fare"].fillna(-1)

    #titanic_df['Age'] = pd.cut(x=titanic_df['Age'], bins=[-2, 0, 21, 25, 40, 80], labels=['unknown', '0-21', '22-25', '26-40', '41-80'])

    titanic_df['Age'] = pd.cut(x=titanic_df['Age'], bins=[-2, 0, 21, 25, 40, 80], labels=['unknown', '0-21', '22-25', '26-40', '41-80']).replace({'0-21':0, '22-25':1, '26-40':2, '41-80':3, 'unknown':4}).astype("int64")
    titanic_df['Sex'] = titanic_df['Sex'].replace({'male': 1, 'female': 0})


    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=["PassengerId"],
        description="Titanic dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
