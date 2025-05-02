import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from datetime import datetime, timedelta

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# defines activity structure
class Activity:
    def __init__(self, timestamp, activity, system, description, user_id, ip_address):
        self.timestamp = timestamp  # activity time
        self.activity = activity    # action taken
        self.system = system        # system involved
        self.description = description  # action details
        self.user_id = user_id      # user who acted
        self.ip_address = ip_address  # network ip

# generates fake activity data
def generate_fake_data(n_samples, anomaly_ratio):
    # activity types
    activities = [
        'User Login', 'File Access', 'Network Connection Attempt', 'Database Query',
        'Configuration Change', 'API Call', 'System Reboot', 'Log Export',
        'User Account Modification', 'Data Export', 'Security Scan', 'Patch Installation'
    ]
    
    # system types
    systems = [
        'Auth Server', 'File Server', 'Web App', 'Database Cluster', 'Firewall',
        'API Gateway', 'Load Balancer', 'Monitoring Server', 'Backup Server', 'DNS Server'
    ]
    
    # normal activity descriptions
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
    
    # suspicious activity descriptions
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
    
    # user ids and ip addresses
    user_ids = [f'user_{i}' for i in range(1, 51)]
    normal_ips = ['192.168.1.10', '192.168.1.20', '172.16.0.50', '10.0.0.100']
    malicious_ips = ['192.168.1.100', '10.0.0.5', '185.230.125.7']
    
    data = []
    session_id = 0  # groups related activities
    
    # create samples
    for i in range(n_samples):
        # anomaly or normal?
        is_anomaly = np.random.random() < anomaly_ratio
        # new session sometimes
        session_id += 1 if np.random.random() < 0.2 else 0
        
        # timestamp with minutes
        timestamp = datetime.now() - timedelta(minutes=np.random.randint(0, 60))
        
        # pick activity and system
        activity = np.random.choice(activities)
        system = np.random.choice(systems)
        
        # use rare but normal pairs sometimes
        if not is_anomaly and np.random.random() < 0.1:
            activity_system_pairs = [
                ('Database Query', 'Firewall'),
                ('API Call', 'Database Cluster'),
                ('User Login', 'Web App')
            ]
            activity, system = activity_system_pairs[np.random.choice(len(activity_system_pairs))]
        
        # pick description
        if is_anomaly:
            description = np.random.choice(anomalous_descriptions)
            # anomalies can happen anytime
            timestamp = timestamp.replace(hour=np.random.randint(0, 24))
        else:
            description = np.random.choice(normal_descriptions)
            # normal during work hours
            timestamp = timestamp.replace(hour=np.random.randint(8, 18))
        
        # assign user
        user_id = np.random.choice(user_ids)
        if is_anomaly and np.random.random() < 0.3:
            user_id = 'unknown_user'
        
        # assign ip for network activities
        ip_address = None
        if activity in ['User Login', 'Network Connection Attempt', 'API Call']:
            if is_anomaly:
                ip_address = np.random.choice(malicious_ips) if np.random.random() < 0.7 else np.random.choice(normal_ips)
            else:
                ip_address = np.random.choice(normal_ips)
        
        # continue session sometimes
        if i > 0 and np.random.random() < 0.3 and not is_anomaly:
            prev_activity = data[-1]
            if prev_activity.activity == 'User Login' and prev_activity.user_id == user_id:
                activity = np.random.choice(['File Access', 'Database Query', 'API Call'])
                system = np.random.choice(systems)
                description = np.random.choice(normal_descriptions)
                timestamp = prev_activity.timestamp + timedelta(minutes=np.random.randint(1, 10))
        
        # save activity
        data.append(Activity(timestamp, activity, system, description, user_id, ip_address))
    
    return data

# handles anomaly detection
class ActivityAnomalyDetector:
    def __init__(self, contamination=0.1, random_state=42, nu=0.1, kernel='rbf', gamma='scale'):
        # features to use
        self.categorical_features = ['activity', 'system', 'description', 'user_id', 'ip_address']
        self.numerical_features = ['hour_of_day', 'minute_of_hour', 'day_of_week', 'description_length']
        
        # preprocess categorical and numerical
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
        
    def preprocess_data(self, data):
        # convert data to dataframe
        try:
            df = pd.DataFrame({
                'timestamp': [d.timestamp for d in data],
                'activity': [d.activity for d in data],
                'system': [d.system for d in data],
                'description': [d.description for d in data],
                'user_id': [d.user_id for d in data],
                'ip_address': [d.ip_address for d in data]
            })
            
            # extract features
            df['hour_of_day'] = df['timestamp'].apply(lambda x: x.hour)
            df['minute_of_hour'] = df['timestamp'].apply(lambda x: x.minute)
            df['day_of_week'] = df['timestamp'].apply(lambda x: x.weekday())
            df['description_length'] = df['description'].apply(len)
            
            logger.info("Data preprocessing completed successfully")
            return df
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
            
    def train(self, data):
        # train both models
        try:
            df = self.preprocess_data(data)
            features = df[self.categorical_features + self.numerical_features]
            
            # train isolation forest
            self.if_pipeline.fit(features)
            logger.info("Isolation Forest training completed successfully")
            
            # train one-class svm
            self.svm_pipeline.fit(features)
            logger.info("One-Class SVM training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise
            
    def predict(self, data):
        # predict anomalies
        try:
            df = self.preprocess_data(data)
            features = df[self.categorical_features + self.numerical_features]
            
            # isolation forest predictions
            if_predictions = self.if_pipeline.predict(features)
            if_scores = self.if_pipeline.named_steps['model'].decision_function(
                self.if_pipeline.named_steps['preprocessor'].transform(features)
            )
            if_normalized_predictions = np.where(if_predictions == -1, 1, 0)
            
            # one-class svm predictions
            svm_predictions = self.svm_pipeline.predict(features)
            svm_scores = self.svm_pipeline.named_steps['model'].decision_function(
                self.svm_pipeline.named_steps['preprocessor'].transform(features)
            )
            svm_normalized_predictions = np.where(svm_predictions == -1, 1, 0)
            
            # flag anomaly if both agree
            combined_predictions = np.where(
                (if_normalized_predictions + svm_normalized_predictions) == 2, 1, 0
            )
            
            # average scores
            combined_scores = (if_scores + svm_scores) / 2
            
            # store results
            result = df.copy()
            result['if_anomaly'] = if_normalized_predictions
            result['if_anomaly_score'] = if_scores
            result['svm_anomaly'] = svm_normalized_predictions
            result['svm_anomaly_score'] = svm_scores
            result['combined_anomaly'] = combined_predictions
            result['combined_anomaly_score'] = combined_scores
            
            anomaly_count = np.sum(combined_predictions == 1)
            logger.info(f"Detected {anomaly_count} anomalies in {len(data)} samples")
            
            return result
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

def main():
    # generate data and detect anomalies
    try:
        # create training and test sets
        train_data = generate_fake_data(n_samples=20000, anomaly_ratio=0.1)
        test_data = generate_fake_data(n_samples=2000, anomaly_ratio=0.1)
        
        # train detector
        detector = ActivityAnomalyDetector(contamination=0.1, nu=0.1)
        detector.train(train_data)
        
        # predict anomalies
        results = detector.predict(test_data)
        
        # show results
        print("\nAnomaly Detection Results:")
        print(f"Total samples: {len(results)}")
        print(f"Detected anomalies (Isolation Forest): {len(results[results['if_anomaly'] == 1])}")
        print(f"Detected anomalies (One-Class SVM): {len(results[results['svm_anomaly'] == 1])}")
        print(f"Detected anomalies (Combined): {len(results[results['combined_anomaly'] == 1])}")
        
        print("\nSample of detected anomalies (Combined):")
        print(results[results['combined_anomaly'] == 1][[
            'activity', 'system', 'description', 'hour_of_day', 
            'if_anomaly_score', 'svm_anomaly_score', 'combined_anomaly_score'
        ]].head())
        
        # save to csv
        results.to_csv('activity_anomaly_results.csv', index=False)
        logger.info("Results saved to activity_anomaly_results.csv")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()