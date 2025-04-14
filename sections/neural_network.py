import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

def neural_network_section():
    st.header("🧠 Neural Network Classifier Explorer")
    st.write("Upload your dataset, set up the neural network parameters, and explore classification results.")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="nn_classifier")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Dataset Snapshot")
        st.dataframe(data.head())

        columns = data.columns.tolist()
        target_column = st.selectbox("Select the target column", columns)
        feature_columns = st.multiselect("Choose feature columns", [col for col in columns if col != target_column])

        if not feature_columns:
            st.warning("Please select at least one feature column.")
            return

        # Data Preprocessing
        if st.checkbox("Remove rows with missing values"):
            data = data.dropna(subset=feature_columns + [target_column])
            st.success("Rows with missing data have been removed.")

        # Auto binning if needed
        if data[target_column].dtype in [np.float64, np.int64] and data[target_column].nunique() > 50:
            data[target_column] = pd.qcut(data[target_column], q=10, labels=False, duplicates='drop')
            st.info("Target column has been automatically binned into 10 classes for classification.")

        X = data[feature_columns].values
        y = data[target_column].values

        # Encoding if labels are non-numeric
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)
        else:
            le = None

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        test_size = st.slider("Select test size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        st.subheader("Set Model Hyperparameters")
        epochs = st.slider("Number of Epochs", 1, 100, 10)
        batch_size = st.slider("Batch Size", 8, 128, 32)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001, format="%.4f")

        num_classes = len(np.unique(y_train))
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        st.subheader("📈 Model Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()

        class TrainingProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{epochs}")

        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[TrainingProgressCallback()]
            )
        except Exception as e:
            st.error(f"Error during training: {e}")
            return

        # Metrics Plotting
        st.subheader("📊 Training Performance Metrics")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title("Loss Curve")
        ax1.legend()

        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title("Accuracy Curve")
        ax2.legend()
        st.pyplot(fig)

        st.success(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")

        # Custom Prediction Section
        st.subheader("🔍 Make a Custom Prediction")
        custom_inputs = {}
        for feature in feature_columns:
            default_val = float(data[feature].mean())
            custom_inputs[feature] = st.number_input(f"Enter value for {feature}", value=default_val)

        if st.button("Make Prediction"):
            custom_df = pd.DataFrame([custom_inputs])
            custom_scaled = scaler.transform(custom_df)
            prediction = model.predict(custom_scaled)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = le.inverse_transform([predicted_class])[0] if le else predicted_class
            st.success(f"Predicted {target_column}: {predicted_label}")
    else:
        st.info("Please upload a classification dataset to begin.")
