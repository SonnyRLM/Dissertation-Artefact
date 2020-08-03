import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Change to True if you wish to re-scramble training/test/eval sets
save_datasets = False  # Save testing / training dataset csv's if necessary

# Open training dataset
train_df = pd.read_csv('data/boneage-training-dataset.csv', index_col=False)

# Count number of males and females in training dataset
train_df['gender'] = train_df['male'].apply(lambda x: '1' if x else '0')


# Display split of genders
sns.countplot(x=train_df['gender'])
plt.title('Gender representation in dataset')
plt.ylabel('Number of samples')
plt.xlabel('Gender (0 = Female | 1 = Male)')
plt.show()

# ------------------ Training data analysis -------------------
# oldest / youngest aged
print('MAX age: ' + str(train_df['boneage'].max()) + ' months')
print('MIN age: ' + str(train_df['boneage'].min()) + ' months')

# mean / median ages
mean_boneage = train_df['boneage'].mean()
print('Mean: ' + str(mean_boneage) + ' months')
print('Median: ' + str(train_df['boneage'].median()) + ' months')

# standard deviation
std_boneage = train_df['boneage'].std()
print("Standard Deviation: " + str(std_boneage) + ' months')

# calculate z-score for training purposes
train_df['bone_age_z'] = (train_df['boneage'] - mean_boneage) / (std_boneage)

# Histogram of bone ages
train_df['boneage'].hist(color='blue')
plt.xlabel('Age (months)')
plt.ylabel('Number of people')
plt.title('Number of people in each age group')

plt.show()

# Age distribution of genders
male = train_df[train_df['gender'] == '1']
female = train_df[train_df['gender'] == '0']

fig, ax = plt.subplots(2,1)
ax[0].hist(male['boneage'], color='blue')
ax[0].set_ylabel('No. of males')

ax[1].hist(female['boneage'], color='red')
ax[1].set_ylabel('No. of females')

ax[0].set_title('Number of people in each age group, split into gender')
fig.set_size_inches((10, 7))

plt.show()


if save_datasets:
    # Split a testing set from the training set
    train_set, test_set = train_test_split(train_df, test_size=0.05, shuffle=True)  # Create testing dataset

    # Take a small subset of the testing set as the evaluation set
    test_set, eval_set = train_test_split(test_set, test_size=0.1, shuffle=True)  # Create small validation dataset

    # Save dataset csv's
    train_set.to_csv('data/training_data.csv', index=False)

    eval_set.to_csv('data/eval_data.csv', index=False)

    test_set.to_csv('data/testing_data.csv', index=False)



