# Implemented by Sertay Akpinar, 06.01.2020
import numpy as np
import pandas as pd
from sklearn import ensemble
import re

# Load the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#feature engineering
full_data = [train, test]
PassengerId = test['PassengerId']
sex_mapping = {'female': 0, 'male': 1}
title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
drop_elements = ['PassengerId', 'Name', 'Ticket', 'SibSp', 'Cabin']
non_common_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']

for dataset in full_data:

    # Create new feature FamilySize as a combination of SibSp and Parch
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Create new feature IsAlone from FamilySize
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    # Create new feature Has_Cabin that tells whether a passenger had a cabin on the Titanic
    dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    # Remove all NULLS in the Embarked column
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    # Remove all NULLS in the Fare column
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    # Remove all NULLS in the Age column
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_random_list

    # Get titles from the names
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

    # Group all non-common titles into a "Rare" title
    dataset['Title'] = dataset['Title'].replace(non_common_titles, 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # Mapping titles
    dataset['Title'] = dataset['Title'].map(title_mapping)

    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

    # Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3

    # Mapping Age
    dataset.loc[dataset['Age'] <= 14, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 14) & (dataset['Age'] <= 25), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 35), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 50), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 64), 'Age'] = 4
    dataset.loc[dataset['Age'] > 64, 'Age'] = 5;

#remove the elements which has no longer containing relevant information
train = train.drop(drop_elements, axis=1)
test = test.drop(drop_elements, axis=1)

#initialize the trainers and the tester
y_train = train['Survived']
x_train = train.drop(['Survived'], axis=1)
x_test = test

#classification of the random forest
forest = ensemble.RandomForestClassifier(min_samples_split=4, max_depth=6, n_estimators=1000)

#predict the result and write it in to a csv file to submit
forest = forest.fit(x_train, y_train)
forest.fit(x_train,y_train)
y_pred = forest.predict(x_test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": y_pred})
submission.to_csv('submission.csv', index=False)