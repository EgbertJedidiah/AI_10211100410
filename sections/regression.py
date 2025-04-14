import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, explained_variance_score

# Function to fetch data from uploaded file
def fetch_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# Function to clean data (exclude non-numeric columns and handle "yes"/"no")
def clean_data(df):
    # Replace "yes" and "no" with 1 and 0
    df = df.applymap(lambda x: 1 if x == 'yes' else (0 if x == 'no' else x))

    # Exclude non-numeric columns (keep only numeric columns)
    df_cleaned = df.select_dtypes(include=['number', 'float64', 'int64'])
    return df_cleaned

# Function to train the linear regression model
def train_linear_model(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    return model, mse, evs

# Main regression section for Streamlit
def regression_section():
    st.title("📉 Regression Studio📈")
    st.markdown("Explore your data with **Linear Regression** and visualize predictions in real-time.")

    uploaded_file = st.file_uploader("📂 Upload a CSV file", type="csv")
    if uploaded_file is not None:
        df = fetch_data(uploaded_file)
        st.write("🔍 Preview of your dataset:")
        st.dataframe(df.head())

        all_columns = df.columns.tolist()
        target = st.selectbox("🎯 Select the target column", all_columns)

        if target and target in df.columns:
            # Only select valid features (exclude target)
            feature_options = [col for col in df.columns if col != target]
            default_features = feature_options.copy()

            features = st.multiselect(
                "🧩 Select feature columns",
                options=feature_options,
                default=default_features
            )

            if features:
                # Clean data (remove non-numeric columns)
                df_cleaned = clean_data(df)

                # Ensure the features selected are in the cleaned dataframe
                valid_features = [feature for feature in features if feature in df_cleaned.columns]

                # Prepare the model dataframe
                df_model = df_cleaned[valid_features + [target]]

                # Separate features and target after cleaning
                X = df_model[valid_features]
                y = df_model[target]

                # Split into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                model, mse, evs = train_linear_model(X_train, y_train, X_test, y_test)

                # Display results
                st.success(f"📉 Mean Squared Error: {mse:.2f}")
                st.info(f"📈 Explained Variance Score: {evs:.2f}")

                # Plot Actual vs Predicted values
                st.subheader("📊 Actual vs Predicted")
                fig, ax = plt.subplots(figsize=(8, 5))
                y_pred = model.predict(X_test)
                ax.scatter(y_test, y_pred, color="#1f77b4", label="Predictions")
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="crimson", linestyle="--", label="Ideal")
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Prediction Accuracy")
                ax.legend()
                st.pyplot(fig)
