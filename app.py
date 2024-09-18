import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Define the pages
PAGES = {
    "EDA": "eda",
    "Prediction": "prediction"
}

def load_data():
    url = 'https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv'
    return pd.read_csv(url)

def eda_page(df):
    st.title("Exploratory Data Analysis")
    st.write("### Data Overview")
    st.dataframe(df)
    st.write("### Dataset Description")
    st.write(df.describe())

    st.write("### Target Variable Distribution")
    fig, ax = plt.subplots()
    df['target'].value_counts().plot(kind="bar", color=["salmon", "lightblue"], ax=ax)
    plt.title("Heart Disease Count")
    st.pyplot(fig)

    st.write("### Missing Values")
    st.write(df.isna().sum())

    categorical_val = [col for col in df.columns if len(df[col].unique()) <= 10]
    continuous_val = [col for col in df.columns if len(df[col].unique()) > 10]

    st.write("### Histograms for Categorical Variables")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for i, column in enumerate(categorical_val):
        df[df["target"] == 0][column].hist(bins=35, color='blue', alpha=0.6, ax=axes[i], label='No Heart Disease')
        df[df["target"] == 1][column].hist(bins=35, color='red', alpha=0.6, ax=axes[i], label='Heart Disease')
        axes[i].legend()
        axes[i].set_xlabel(column)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("### Histograms for Continuous Variables")
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()
    for i, column in enumerate(continuous_val):
        df[df["target"] == 0][column].hist(bins=35, color='blue', alpha=0.6, ax=axes[i], label='No Heart Disease')
        df[df["target"] == 1][column].hist(bins=35, color='red', alpha=0.6, ax=axes[i], label='Heart Disease')
        axes[i].legend()
        axes[i].set_xlabel(column)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("### Scatter Plot: Age vs Max Heart Rate")
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.scatter(df.age[df.target == 1], df.thalach[df.target == 1], c="salmon", label="Disease")
    plt.scatter(df.age[df.target == 0], df.thalach[df.target == 0], c="lightblue", label="No Disease")
    plt.title("Heart Disease in function of Age and Max Heart Rate")
    plt.xlabel("Age")
    plt.ylabel("Max Heart Rate")
    plt.legend()
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    st.write("### Correlation with Target")
    fig, ax = plt.subplots(figsize=(12, 8))
    df.drop('target', axis=1).corrwith(df.target).plot(kind='bar', grid=True, ax=ax)
    plt.title("Correlation with Target")
    st.pyplot(fig)

def prediction_page(df):
    st.title("Prediction")
    
    # Prepare dataset for prediction
    categorical_val = [col for col in df.columns if len(df[col].unique()) <= 10]
    categorical_val.remove('target')
    dataset = pd.get_dummies(df, columns=categorical_val)
    scaler = StandardScaler()
    cols_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    dataset[cols_to_scale] = scaler.fit_transform(dataset[cols_to_scale])

    # Train-test split
    X = dataset.drop('target', axis=1)
    y = dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Logistic Regression model
    st.write("### Logistic Regression Model")

    # Train the model
    lr_clf = LogisticRegression(solver='liblinear')
    lr_clf.fit(X_train, y_train)

    # Function to display scores
    def print_score(clf, X_train, y_train, X_test, y_test, train=True):
        if train:
            pred = clf.predict(X_train)
            clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
            st.write(f"### Train Result")
            st.write(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
            st.write(f"Confusion Matrix:\n {confusion_matrix(y_train, pred)}")
            st.write(f"Classification Report:\n", clf_report)

        else:
            pred = clf.predict(X_test)
            clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
            st.write(f"### Test Result")
            st.write(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
            st.write(f"Confusion Matrix:\n {confusion_matrix(y_test, pred)}")
            st.write(f"Classification Report:\n", clf_report)

    # Display the results
    print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)

    # Summary results dataframe
    train_score = accuracy_score(y_train, lr_clf.predict(X_train)) * 100
    test_score = accuracy_score(y_test, lr_clf.predict(X_test)) * 100
    results_df = pd.DataFrame([["Logistic Regression", train_score, test_score]],
                              columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

    st.write("### Model Performance Summary")
    st.dataframe(results_df)

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    
    # Load data
    df = load_data()
    
    # Load the selected page
    if selection == "EDA":
        eda_page(df)
    elif selection == "Prediction":
        prediction_page(df)

if __name__ == "__main__":
    main()

