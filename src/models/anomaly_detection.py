import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress tensorflow info messages
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging
from datetime import datetime, timedelta

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# define activity structure
class Activity:
    def __init__(self, timestamp, activity, system, description, user_id, ip_address, is_anomaly):
        self.timestamp = timestamp
        self.activity = activity
        self.system = system
        self.description = description
        self.user_id = user_id
        self.ip_address = ip_address
        self.is_anomaly = is_anomaly  # track true anomaly status

# generate fake activity data with temporal structure
def generate_fake_data(n_samples, anomaly_ratio):
    activities = [
        'User Login', 'File Access', 'Network Connection Attempt', 'Database Query',
        'Configuration Change', 'API Call', 'System Reboot', 'Log Export',
        'User Account Modification', 'Data Export', 'Security Scan', 'Patch Installation'
    ]
    systems = [
        'Auth Server', 'File Server', 'Web App', 'Database Cluster', 'Firewall',
        'API Gateway', 'Load Balancer', 'Monitoring Server', 'Backup Server', 'DNS Server'
    ]
    normal_descriptions = [
        'Successful login from known device',
        'Read access to configuration file',
        'Standard API health check received',
        'User password reset completed',
        'Configuration backup process initiated',
        'Performed SELECT query on public table',
        'System health check completed',
        'Exported logs for audit',
        'Added new user account',
        'Data export to analytics platform',
        'Scheduled security scan completed',
        'Applied security patch successfully'
    ]
    anomalous_descriptions = [
        'Multiple failed login attempts from IP 192.168.1.100',
        'Accessed sensitive financial_report.xlsx',
        'Attempted connection to known malicious IP 10.0.0.5',
        'Executed DROP TABLE command on users table',
        'Disabled firewall rule #3',
        'Anomalous high volume of requests to /admin endpoint',
        'Unauthorized system reboot initiated',
        'Exported logs to external server',
        'Modified admin account permissions',
        'Large data export without approval',
        'Security scan triggered outside schedule',
        'Failed patch installation with errors'
    ]
    user_ids = [f'user_{i}' for i in range(1, 51)]
    normal_ips = ['192.168.1.10', '192.168.1.20', '172.16.0.50', '10.0.0.100']
    malicious_ips = ['192.168.1.100', '10.0.0.5', '185.230.125.7']
    
    data = []
    base_time = datetime.now() - timedelta(days=30)  # start 30 days ago
    session_id = 0
    user_sessions = {}  # track user sessions
    
    for i in range(n_samples):
        is_anomaly = np.random.random() < anomaly_ratio
        session_id += 1 if np.random.random() < 0.2 else 0
        
        # generate timestamp
        if i == 0:
            timestamp = base_time
        else:
            timestamp = data[-1].timestamp + timedelta(minutes=np.random.randint(1, 5))
        
        # select user
        user_id = np.random.choice(user_ids)
        if is_anomaly and np.random.random() < 0.3:
            user_id = 'unknown_user'
        
        # manage user session
        if user_id not in user_sessions or np.random.random() < 0.3:
            user_sessions[user_id] = {
                'last_activity': None,
                'activity_count': 0,
                'session_start': timestamp
            }
        
        # select activity and system
        activity = np.random.choice(activities)
        system = np.random.choice(systems)
        
        # normal session behavior
        if not is_anomaly:
            if user_sessions[user_id]['activity_count'] == 0:
                activity = 'User Login'
                system = 'Auth Server'
            elif user_sessions[user_id]['activity_count'] == 1:
                activity = np.random.choice(['File Access', 'Database Query', 'API Call'])
                system = np.random.choice(['File Server', 'Database Cluster', 'API Gateway'])
            else:
                activity = np.random.choice(activities)
                system = np.random.choice(systems)
            # normal activities during business hours
            hour = np.random.randint(8, 18)
            timestamp = timestamp.replace(hour=hour, minute=np.random.randint(0, 60))
            description = np.random.choice(normal_descriptions)
        else:
            # anomalous behavior
            description = np.random.choice(anomalous_descriptions)
            timestamp = timestamp.replace(hour=np.random.randint(0, 24))
            if np.random.random() < 0.5:
                activity = np.random.choice(['System Reboot', 'User Account Modification', 'Data Export'])
                system = np.random.choice(['Auth Server', 'Database Cluster', 'Backup Server'])
        
        # assign ip address
        ip_address = None
        if activity in ['User Login', 'Network Connection Attempt', 'API Call']:
            if is_anomaly:
                ip_address = np.random.choice(malicious_ips) if np.random.random() < 0.7 else np.random.choice(normal_ips)
            else:
                ip_address = np.random.choice(normal_ips)
        
        # update session
        user_sessions[user_id]['last_activity'] = activity
        user_sessions[user_id]['activity_count'] += 1
        if user_sessions[user_id]['activity_count'] >= np.random.randint(3, 6):
            user_sessions.pop(user_id)
        
        data.append(Activity(timestamp, activity, system, description, user_id, ip_address, is_anomaly))
    
    # sort by timestamp
    data.sort(key=lambda x: x.timestamp)
    return data

# handle anomaly detection
class ActivityAnomalyDetector:
    def __init__(self, contamination=0.1, random_state=42, nu=0.1, kernel='rbf', gamma='scale', sequence_length=5):
        self.categorical_features = ['activity', 'system', 'description', 'user_id', 'ip_address']
        self.numerical_features = ['hour_of_day', 'minute_of_hour', 'day_of_week', 'description_length', 'time_interval']
        self.sequence_length = sequence_length
        # define preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), self.categorical_features),
                ('num', StandardScaler(), self.numerical_features)
            ]
        )
        # isolation forest pipeline
        self.if_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=100,
                max_samples='auto'
            ))
        ])
        # one-class svm pipeline
        self.svm_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', OneClassSVM(
                nu=nu,
                kernel=kernel,
                gamma=gamma
            ))
        ])
        self.lstm_model = None  # defer model creation until training
    
    # build lstm model with specified feature dimension
    def _build_lstm_model(self, feature_dim):
        model = Sequential([
            LSTM(32, input_shape=(self.sequence_length, feature_dim), return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    # preprocess data
    def preprocess_data(self, data, for_lstm=False):
        try:
            df = pd.DataFrame({
                'timestamp': [d.timestamp for d in data],
                'activity': [d.activity for d in data],
                'system': [d.system for d in data],
                'description': [d.description for d in data],
                'user_id': [d.user_id for d in data],
                'ip_address': [d.ip_address for d in data],
                'is_anomaly': [d.is_anomaly for d in data]
            })
            # extract temporal features
            df['hour_of_day'] = df['timestamp'].apply(lambda x: x.hour)
            df['minute_of_hour'] = df['timestamp'].apply(lambda x: x.minute)
            df['day_of_week'] = df['timestamp'].apply(lambda x: x.weekday())
            df['description_length'] = df['description'].apply(len)
            # add time interval feature
            df['time_interval'] = df['timestamp'].diff().dt.total_seconds().fillna(0) / 60.0
            
            if for_lstm:
                # sort for lstm sequences
                df = df.sort_values('timestamp')
                features = self.categorical_features + self.numerical_features
                X = self.preprocessor.fit_transform(df[features]).toarray()
                y = df['is_anomaly'].values
                # create sequences
                X_seq, y_seq = [], []
                for i in range(len(X) - self.sequence_length + 1):
                    X_seq.append(X[i:i + self.sequence_length])
                    y_seq.append(y[i + self.sequence_length - 1])
                return np.array(X_seq), np.array(y_seq), df
            else:
                logger.info("Data preprocessing completed successfully")
                return df
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    # train models
    def train(self, data):
        try:
            df = self.preprocess_data(data)
            features = df[self.categorical_features + self.numerical_features]
            # train isolation forest
            self.if_pipeline.fit(features)
            logger.info("Isolation Forest training completed successfully")
            # train svm
            self.svm_pipeline.fit(features)
            logger.info("One-Class SVM training completed successfully")
            # train lstm
            X_seq, y_seq, _ = self.preprocess_data(data, for_lstm=True)
            if len(X_seq) > 0:
                # build lstm model with correct feature dimension
                feature_dim = X_seq.shape[2]  # number of features after preprocessing
                self.lstm_model = self._build_lstm_model(feature_dim)
                self.lstm_model.fit(X_seq, y_seq, epochs=10, batch_size=32, verbose=0)
                logger.info("LSTM training completed successfully")
            else:
                logger.warning("Insufficient data for LSTM training")
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise
    
    # predict anomalies
    def predict(self, data):
        try:
            df = self.preprocess_data(data)
            features = df[self.categorical_features + self.numerical_features]
            
            # isolation forest predictions
            if_predictions = self.if_pipeline.predict(features)
            if_scores = self.if_pipeline.named_steps['model'].decision_function(
                self.if_pipeline.named_steps['preprocessor'].transform(features)
            )
            if_normalized_predictions = np.where(if_predictions == -1, 1, 0)
            
            # svm predictions
            svm_predictions = self.svm_pipeline.predict(features)
            svm_scores = self.svm_pipeline.named_steps['model'].decision_function(
                self.svm_pipeline.named_steps['preprocessor'].transform(features)
            )
            svm_normalized_predictions = np.where(svm_predictions == -1, 1, 0)
            
            # lstm predictions
            X_seq, _, full_df = self.preprocess_data(data, for_lstm=True)
            lstm_predictions = np.zeros(len(df))
            lstm_scores = np.zeros(len(df))
            if len(X_seq) > 0 and self.lstm_model is not None:
                lstm_pred = self.lstm_model.predict(X_seq, verbose=0)
                lstm_predictions[-len(lstm_pred):] = (lstm_pred > 0.5).astype(int).flatten()
                lstm_scores[-len(lstm_pred):] = lstm_pred.flatten()
            
            # combine predictions
            combined_predictions = np.where(
                (if_normalized_predictions + svm_normalized_predictions + lstm_predictions) >= 2, 1, 0
            )
            combined_scores = (if_scores + svm_scores + lstm_scores) / 3
            
            # create result dataframe
            result = df.copy()
            result['if_anomaly'] = if_normalized_predictions
            result['if_anomaly_score'] = if_scores
            result['svm_anomaly'] = svm_normalized_predictions
            result['svm_anomaly_score'] = svm_scores
            result['lstm_anomaly'] = lstm_predictions
            result['lstm_anomaly_score'] = lstm_scores
            result['combined_anomaly'] = combined_predictions
            result['combined_anomaly_score'] = combined_scores
            
            anomaly_count = np.sum(combined_predictions == 1)
            logger.info(f"Detected {anomaly_count} anomalies in {len(data)} samples")
            return result
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

# main execution
def main():
    try:
        # generate data
        train_data = generate_fake_data(n_samples=10000, anomaly_ratio=0.1)  # reduced for cpu
        test_data = generate_fake_data(n_samples=2000, anomaly_ratio=0.1)
        
        # initialize detector
        detector = ActivityAnomalyDetector(contamination=0.1, nu=0.1, sequence_length=5)  # shorter sequence for cpu
        detector.train(train_data)
        
        # predict anomalies
        results = detector.predict(test_data)
        
        # calculate accuracy
        true_labels = results['is_anomaly'].astype(int)
        predicted_labels = results['combined_anomaly']
        accuracy = accuracy_score(true_labels, predicted_labels) * 100
        
        # display results
        print("\nAnomaly Detection Results:")
        print(f"Total samples: {len(results)}")
        print(f"Detected anomalies (Isolation Forest): {len(results[results['if_anomaly'] == 1])}")
        print(f"Detected anomalies (One-Class SVM): {len(results[results['svm_anomaly'] == 1])}")
        print(f"Detected anomalies (LSTM): {len(results[results['lstm_anomaly'] == 1])}")
        print(f"Detected anomalies (Combined): {len(results[results['combined_anomaly'] == 1])}")
        print(f"Accuracy of Combined System: {accuracy:.1f}%")
 
        print("\nSample of detected anomalies (Combined):")
        print(results[results['combined_anomaly'] == 1][[
            'activity', 'system', 'description', 'hour_of_day', 
            'if_anomaly_score', 'svm_anomaly_score', 'lstm_anomaly_score', 'combined_anomaly_score'
        ]].head())
        
        # save results
        results.to_csv('activity_anomaly_results.csv', index=False)
        logger.info("Results saved to activity_anomaly_results.csv")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()