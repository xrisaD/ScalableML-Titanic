import os
import modal
    
BACKFILL=False
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_passenger(survived, pclass_max, pclass_min, sex_max, sex_min,
                    age_max, age_min, fare_max, fare_min):
    """
    Returns a single passanger as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ "pclass": [int(random.uniform(pclass_max, pclass_min))],
                       "sex": [random.randint(sex_min, sex_max)],
                       "age": [random.randint(age_min, age_max)],
                       "fare": [random.uniform(fare_max, fare_min)]
                      })
    df['survived'] = survived
    return df


def get_random_passenger():
    """
    Returns a DataFrame containing one random passenger
    """
    import pandas as pd
    import random

    # randomly pick one of these 2 and write it to the featurestore
    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        passenger_df = generate_passenger(1, 4, 1, 2, 0, 5, 0, 500, 0)
        print("Survived added")
    else:
        passenger_df = generate_passenger(0, 4, 1, 2, 0, 5, 0, 500, 0)
        print("Not survived added")


    return passenger_df



def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()
    #
    # if BACKFILL == True:
    #     titanic_df = pd.read_csv("https://repo.hops.works/master/hopsworks-tutorials/data/iris.csv")
    # else:

    titanic_df = get_random_passenger()

    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=["passengerid"],
        description="Titanic dataset")

    # find next id
    #query.filter(titanic_fg
    #feature_view = fs.get_feature_view(name="titanic_modal", version=1)

    max = titanic_fg.read()["passengerid"].max()
    titanic_df["passengerid"] = max+1

    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
