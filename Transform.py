import pandas as pd

df = pd.read_csv("iris.csv", header=None)
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df.columns = columns
print(df)

new_df = df.replace({'class': [ "Iris-setosa", "Iris-versicolor", "Iris-virginica"]}, 
                                   {'class': ["1 0 0", "0 1 0", "0 0 1"]})

new_df.to_csv("new_iris.csv", index=False)

print(new_df)

