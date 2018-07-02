import pandas as pd
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Combining test and train files
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index=True, sort=True)

# Calculate the average weight of items:
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')

# Create a variable for missing values
missing = data['Item_Weight'].isnull()

# Fill the missing values
print('Orignal #missing: %d' % sum(missing))
data.loc[missing, "Item_Weight"] = data.loc[missing, "Item_Identifier"].apply(lambda x: item_avg_weight.loc[x].values[0])
print('Final #missing: %d' % sum(data['Item_Weight'].isnull()))

# Calculate average visibility of a product
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
print('Number of 0 values initially: %d' % sum(missing))
data.loc[missing, 'Item_Visibility'] = data.loc[missing, 'Item_Identifier'].apply(lambda x: visibility_avg.loc[x].values[0])
print('Number of 0 values after modification: %d' % sum(data['Item_Visibility'] == 0))

# Calculate the mean sales by type:
data.pivot_table(values='Item_Outlet_Sales', index='Outlet_Type')

# Replace zero values with mean visibility of a product:
missing = (data['Item_Visibility'] == 0)
data.loc[missing, 'Item_Visibility'] = data.loc[missing, 'Item_Identifier'].apply(lambda x: visibility_avg[x])

# Create three new, broader categories:
data['Item_Identifier'].value_counts()
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'})
data['Item_Type_Combined'].value_counts()

# Determine years of operation of outlets
# The data comes from year 2013
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()

# Correct typos in "fat" category
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})

# Create new category for products that are not food but have fat content specified
data.loc[data['Item_Type_Combined'] == "Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()

# New outlet identifier variable
le = LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Item_Type_Combined', 'Outlet_Type', 'Outlet']
data[var_mod] = data[var_mod].astype('str')
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])

# Introducing dummy variables - one hot coding:
data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type',
                                     'Item_Type_Combined', 'Outlet'])

# Drop the columns which have been converted to different types:
data.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

# Split the dataset into train and test again:
train = data.loc[data['source'] == "train"]
test = data.loc[data['source'] == "test"]

# Drop extra columns that wont be used:
test.drop(['Item_Outlet_Sales', 'source'], axis=1, inplace=True)
train.drop(['source'], axis=1, inplace=True)

# Export the results:
train.to_csv("trainClean.csv", index=False)
test.to_csv("testClean.csv", index=False)
