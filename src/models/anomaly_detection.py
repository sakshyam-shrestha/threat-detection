import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Activity:
    """Class to represent an activity log entry with threat severity."""
    def __init__(self, timestamp, activity, system, description, user_id, ip_address, is_anomaly, severity=None):
        self.timestamp = timestamp
        self.activity = activity
        self.system = system
        self.description = description
        self.user_id = user_id
        self.ip_address = ip_address
        self.is_anomaly = is_anomaly
        self.severity = severity

def generate_fake_data(n_samples, anomaly_ratio):
    """Generate synthetic activity data with increased variety and controlled anomalies and severity levels."""
    try:
        activities = [
            'User Login', 'File Access', 'Database Query', 'API Call', 'Network Ping', 
            'System Update', 'Email Sent', 'Configuration Change', 'Session Timeout', 
            'Data Backup'
        ]
        systems = [
            'Auth Server', 'File Server', 'Database Cluster', 'API Gateway', 'Web Server', 
            'Backup Server', 'Monitoring System', 'Firewall', 'Load Balancer', 'Logging System'
        ]
        normal_descriptions = [
            'Successful login', 'Read access to file', 'SELECT query', 'API health check', 
            'Routine system check', 'File download', 'Email delivery', 'Configuration saved', 
            'Backup completed', 'Session ended'
        ]
        anomalous_descriptions = {
            'Low': [
                'Multiple failed logins', 'Unusual login time', 'Suspicious port scan', 
                'Failed API authentication', 'Unexpected session duration'
            ],
            'Medium': [
                'Accessed sensitive file', 'High-volume API requests', 'Unusual data export', 
                'Repeated file modifications', 'Unexpected configuration change'
            ],
            'High': [
                'DROP TABLE command', 'Unauthorized access attempt', 'Privilege escalation attempt', 
                'Malware detected', 'Data exfiltration attempt'
            ]
        }
        user_ids = [f'user_{i}' for i in range(1, 101)]  # Increased range
        ip_addresses = (
            [f'192.168.1.{i}' for i in range(10, 100)] + 
            [f'10.0.0.{i}' for i in range(1, 20)] + 
            ['192.168.1.100', '10.0.0.5', '172.16.0.1']
        )
        severity_levels = ['Low', 'Medium', 'High']

        data = []
        base_time = datetime.now() - timedelta(days=30)
        for i in range(n_samples):
            is_anomaly = np.random.random() < anomaly_ratio
            timestamp = base_time + timedelta(minutes=i * 5 + np.random.randint(-2, 3))
            activity = np.random.choice(activities)
            system = np.random.choice(systems)
            if is_anomaly:
                severity = np.random.choice(severity_levels, p=[0.5, 0.3, 0.2])
                description = np.random.choice(anomalous_descriptions[severity])
            else:
                severity = None
                description = np.random.choice(normal_descriptions)
            user_id = np.random.choice(user_ids)
            ip_address = np.random.choice(ip_addresses, p=[0.95/len(ip_addresses[:-3])] * (len(ip_addresses)-3) + [0.05/3] * 3)
            data.append(Activity(timestamp, activity, system, description, user_id, ip_address, is_anomaly, severity))
        
        data.sort(key=lambda x: x.timestamp)
        logger.info(f"Generated {n_samples} data points with approximately {anomaly_ratio*100}% anomalies")
        return data
    except Exception as e:
        logger.error(f"Failed to generate data: {e}")
        raise

class ActivityAnomalyDetector:
    """Class for detecting and classifying anomalies using multiple algorithms and Random Forest."""
    def __init__(self, contamination=0.1, random_state=42, nu=0.1, sequence_length=5):
        self.categorical_features = ['activity', 'system', 'description', 'user_id', 'ip_address']
        self.numerical_features = ['hour_of_day', 'description_length', 'day_of_week', 'is_weekend']
        self.sequence_length = sequence_length
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=True), self.categorical_features),
                ('num', StandardScaler(), self.numerical_features)
            ])
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100)
        self.one_class_svm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        self.lstm_model = None
        self.autoencoder = None
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
        self.is_fitted = False

    def _build_lstm_model(self, feature_dim):
        """Build LSTM model for sequence-based anomaly detection."""
        model = Sequential([
            LSTM(32, input_shape=(self.sequence_length, feature_dim), return_sequences=True),
            LSTM(16, return_sequences=False),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_autoencoder(self, feature_dim):
        """Build Autoencoder for reconstruction-based anomaly detection."""
        model = Sequential([
            Dense(16, activation='relu', input_dim=feature_dim),
            Dense(8, activation='relu'),
            Dense(16, activation='relu'),
            Dense(feature_dim, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
        return model

    def _create_dataframe(self, data):
        """Convert activity objects to a DataFrame with engineered features."""
        try:
            df = pd.DataFrame({
                'timestamp': [d.timestamp for d in data],
                'activity': [d.activity for d in data],
                'system': [d.system for d in data],
                'description': [d.description for d in data],
                'user_id': [d.user_id for d in data],
                'ip_address': [d.ip_address for d in data],
                'is_anomaly': [d.is_anomaly for d in data],
                'severity': [d.severity for d in data]
            })
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['description_length'] = df['description'].str.len() / 100.0  # Normalize
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            return df
        except Exception as e:
            logger.error(f"Failed to create DataFrame: {e}")
            raise

    def _prepare_lstm_sequences(self, X, y=None):
        """Prepare sequence data for LSTM."""
        try:
            X_seq = []
            y_seq = []
            for i in range(len(X) - self.sequence_length + 1):
                X_seq.append(X[i:i + self.sequence_length])
                if y is not None:
                    y_seq.append(y[i + self.sequence_length - 1])
            return np.array(X_seq), np.array(y_seq) if y is not None else None
        except Exception as e:
            logger.error(f"Failed to prepare LSTM sequences: {e}")
            raise

    def train(self, data):
        """Train all anomaly detection models and Random Forest classifier."""
        try:
            df = self._create_dataframe(data)
            X = self.preprocessor.fit_transform(df[self.categorical_features + self.numerical_features]).toarray()
            
            # Train Isolation Forest
            self.isolation_forest.fit(X)
            logger.info("Isolation Forest trained")

            # Train One-Class SVM
            self.one_class_svm.fit(X)
            logger.info("One-Class SVM trained")

            # Train LSTM
            y = df['is_anomaly'].values
            X_seq, y_seq = self._prepare_lstm_sequences(X, y)
            if len(X_seq) > 0:
                self.lstm_model = self._build_lstm_model(X_seq.shape[2])
                self.lstm_model.fit(X_seq, y_seq, epochs=10, batch_size=32, verbose=0)
                logger.info("LSTM trained")
            else:
                logger.warning("Insufficient data for LSTM training")

            # Train Autoencoder
            normal_data_mask = df['is_anomaly'] == 0
            if normal_data_mask.sum() > 0:
                X_normal = X[normal_data_mask]
                self.autoencoder = self._build_autoencoder(X_normal.shape[1])
                self.autoencoder.fit(X_normal, X_normal, epochs=5, batch_size=32, verbose=0)
                logger.info("Autoencoder trained")
            else:
                logger.warning("Insufficient normal data for Autoencoder")

            # Train Random Forest for severity classification
            anomaly_mask = df['is_anomaly'] == 1
            if anomaly_mask.sum() > 0:
                X_anomalies = X[anomaly_mask]
                y_severity = df.loc[anomaly_mask, 'severity']
                self.random_forest.fit(X_anomalies, y_severity)
                logger.info("Random Forest classifier trained for severity")
            else:
                logger.warning("No anomalies in training data for Random Forest")

            self.is_fitted = True
            logger.info("All models trained successfully")
        except ValueError as e:
            logger.error(f"Training failed due to invalid data: {e}")
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def predict(self, data):
        """Predict anomalies and classify their severity using Random Forest."""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be trained before prediction")
            df = self._create_dataframe(data)
            X = self.preprocessor.transform(df[self.categorical_features + self.numerical_features]).toarray()

            # Isolation Forest predictions
            if_predictions = self.isolation_forest.predict(X)
            if_scores = self.isolation_forest.decision_function(X)
            if_anomalies = np.where(if_predictions == -1, 1, 0)
            if_weights = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)

            # One-Class SVM predictions
            svm_predictions = self.one_class_svm.predict(X)
            svm_scores = self.one_class_svm.decision_function(X)
            svm_anomalies = np.where(svm_predictions == -1, 1, 0)
            svm_weights = (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min() + 1e-10)

            # LSTM predictions
            lstm_anomalies = np.zeros(len(X))
            lstm_scores = np.zeros(len(X))
            lstm_weights = np.zeros(len(X))
            if self.lstm_model is not None:
                X_seq, _ = self._prepare_lstm_sequences(X)
                if len(X_seq) > 0:
                    lstm_pred = self.lstm_model.predict(X_seq, verbose=0)
                    offset = len(X) - len(lstm_pred)
                    lstm_anomalies[offset:] = (lstm_pred > 0.5).astype(int).flatten()
                    lstm_scores[offset:] = lstm_pred.flatten()
                    lstm_weights[offset:] = lstm_pred.flatten()

            # Autoencoder predictions
            autoencoder_anomalies = np.zeros(len(X))
            autoencoder_scores = np.zeros(len(X))
            autoencoder_weights = np.zeros(len(X))
            if self.autoencoder is not None:
                reconstructed = self.autoencoder.predict(X, verbose=0)
                mse = np.mean(np.square(X - reconstructed), axis=1)
                threshold = np.percentile(mse, 95)
                autoencoder_anomalies = (mse > threshold).astype(int)
                autoencoder_scores = mse
                autoencoder_weights = (mse - mse.min()) / (mse.max() - mse.min() + 1e-10)

            # Combine predictions (weighted voting)
            combined_scores = (if_weights + svm_weights + lstm_weights + autoencoder_weights) / 4
            combined_anomalies = np.where(combined_scores > 0.5, 1, 0)

            # Classify severity only for flagged anomalies using Random Forest
            severity_predictions = np.array(['None'] * len(X), dtype=object)
            if combined_anomalies.sum() > 0:
                X_anomalies = X[combined_anomalies == 1]
                severity_predictions[combined_anomalies == 1] = self.random_forest.predict(X_anomalies)
                logger.info(f"Classified severity for {combined_anomalies.sum()} flagged anomalies")
            else:
                logger.info("No anomalies flagged for severity classification")

            # Create result DataFrame
            result = df.assign(
                if_anomaly=if_anomalies,
                if_anomaly_score=if_scores,
                svm_anomaly=svm_anomalies,
                svm_anomaly_score=svm_scores,
                lstm_anomaly=lstm_anomalies,
                lstm_anomaly_score=lstm_scores,
                autoencoder_anomaly=autoencoder_anomalies,
                autoencoder_anomaly_score=autoencoder_scores,
                combined_anomaly=combined_anomalies,
                combined_anomaly_score=combined_scores,
                predicted_severity=severity_predictions
            )
            logger.info(f"Detected {combined_anomalies.sum()} anomalies")
            return result
        except ValueError as e:
            logger.error(f"Prediction failed due to invalid data: {e}")
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

def main():
    """Main function to run the anomaly detection and threat classification pipeline."""
    try:
        # Generate data
        train_data = generate_fake_data(n_samples=10000, anomaly_ratio=0.1)
        test_data = generate_fake_data(n_samples=3000, anomaly_ratio=0.1)

        # Train and predict
        detector = ActivityAnomalyDetector(contamination=0.1, nu=0.1, sequence_length=5)
        detector.train(train_data)
        results = detector.predict(test_data)

        # Evaluate anomaly detection
        true_labels = results['is_anomaly'].astype(int)
        predicted_labels = results['combined_anomaly']
        anomaly_accuracy = accuracy_score(true_labels, predicted_labels) * 100

        # Evaluate severity classification
        anomaly_results = results[results['combined_anomaly'] == 1]
        true_severity = anomaly_results['severity']
        predicted_severity = anomaly_results['predicted_severity']
        if len(anomaly_results) > 0:
            # Filter out None values from true_severity
            valid_mask = true_severity.notna()
            if valid_mask.sum() > 0:
                true_severity_valid = true_severity[valid_mask]
                predicted_severity_valid = predicted_severity[valid_mask]
                severity_report = classification_report(true_severity_valid, predicted_severity_valid, target_names=['Low', 'Medium', 'High'])
            else:
                severity_report = "No valid severity labels for classification."
        else:
            severity_report = "No anomalies detected for severity classification."

        # Output results
        print("\nAnomaly Detection Results:")
        print(f"Total samples: {len(results)}")
        print(f"Detected anomalies (Isolation Forest): {len(results[results['if_anomaly'] == 1])}")
        print(f"Detected anomalies (One-Class SVM): {len(results[results['svm_anomaly'] == 1])}")
        print(f"Detected anomalies (LSTM): {len(results[results['lstm_anomaly'] == 1])}")
        print(f"Detected anomalies (Autoencoder): {len(results[results['autoencoder_anomaly'] == 1])}")
        print(f"Detected anomalies (Combined): {len(results[results['combined_anomaly'] == 1])}")
        print(f"Anomaly Detection Accuracy: {anomaly_accuracy:.1f}%")
        print("\nSeverity Classification Report:")
        print(severity_report)
        print("\nDetected anomalies with true and predicted severity:")
        if len(anomaly_results) > 0:
            print(anomaly_results[[
                'activity', 'system', 'description', 'hour_of_day',
                'combined_anomaly_score', 'severity', 'predicted_severity'
            ]])
        else:
            print("No anomalies detected.")

        # Save results, including severity
        logger.info(f"Saving results with columns: {list(results.columns)}")
        results.to_csv('threat_detection_results.csv', index=False)
        logger.info("Results saved to threat_detection_results.csv")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise
    finally:
        logger.info("Execution completed")

if __name__ == "__main__":
    main()