import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    return df

df = load_data()

# Title
st.title("Titanic Dataset Analysis")

# Show raw data
if st.checkbox("Show raw data"):
    st.write(df.head())

# Check missing values
if st.checkbox("Show missing values"):
    st.write(df.isnull().sum())

# Summary statistics
if st.checkbox("Show summary statistics"):
    st.write(df.describe())

# Data types and missing values
if st.checkbox("Show data info"):
    buffer = []
    df.info(buf=buffer.append)
    st.text("\n".join(buffer))

# Survival Rate by Gender
st.subheader("Survival Rate by Gender")
fig, ax = plt.subplots()
sns.barplot(x="Sex", y="Survived", data=df, ax=ax)
ax.set_title("Survival Rate by Gender")
st.pyplot(fig)

# Survival Rate by Age Group
df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 50, 80], labels=["Child", "Teen", "Young Adult", "Middle Age", "Senior"])
st.subheader("Survival Rate by Age Group")
fig, ax = plt.subplots()
sns.barplot(x="AgeGroup", y="Survived", data=df, ax=ax)
ax.set_title("Survival Rate by Age Group")
st.pyplot(fig)

# Survival Rate by Passenger Class
st.subheader("Survival Rate by Passenger Class")
fig, ax = plt.subplots()
sns.barplot(x="Pclass", y="Survived", data=df, ax=ax)
ax.set_title("Survival Rate by Passenger Class")
st.pyplot(fig)

# Handle Missing Values
df.loc[:, "Age"] = df["Age"].fillna(df["Age"].median())  # Fix chained assignment issue
df.drop(columns=["Cabin"], inplace=True)
df.loc[:, "Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  # Fix chained assignment issue

# Encode Categorical Variables
df.loc[:, "Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Feature Engineering
df.loc[:, "FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Save Cleaned Data
if st.button("Save Cleaned Data"):
    df.to_csv("cleaned_titanic.csv", index=False)
    st.success("Cleaned dataset saved as cleaned_titanic.csv")
