# src/streaming_fraud_detector.py

import json
from kafka import KafkaConsumer, KafkaProducer
import joblib
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RealTimeFraudDetector:
    """
    Handles real-time fraud detection by consuming transaction data from a Kafka topic,
    making predictions using a pre-trained model, and publishing fraud alerts.
    """

    def __init__(self, model_path: str, kafka_config: dict):
        """
        Initializes the real-time detector.
        
        Args:
            model_path (str): Path to the trained fraud detection model.
            kafka_config (dict): Configuration for Kafka consumer and producer.
        """
        self.model = self._load_model(model_path)
        self.kafka_config = kafka_config
        self.consumer = self._create_kafka_consumer()
        self.producer = self._create_kafka_producer()
        self.scaler = joblib.load('models/scaler.joblib') # Assuming you save the scaler

    def _load_model(self, model_path: str):
        """Loads the machine learning model from the specified path."""
        try:
            logging.info("Loading model from %s", model_path)
            return joblib.load(model_path)
        except FileNotFoundError:
            logging.error("Model file not found at %s. Aborting.", model_path)
            raise
    
    def _create_kafka_consumer(self):
        """Creates and returns a Kafka consumer."""
        try:
            return KafkaConsumer(
                self.kafka_config['transaction_topic'],
                bootstrap_servers=self.kafka_config['bootstrap_servers'],
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
        except Exception as e:
            logging.error("Failed to create Kafka consumer: %s", e)
            return None

    def _create_kafka_producer(self):
        """Creates and returns a Kafka producer."""
        try:
            return KafkaProducer(
                bootstrap_servers=self.kafka_config['bootstrap_servers'],
                value_serializer=lambda m: json.dumps(m).encode('utf-8')
            )
        except Exception as e:
            logging.error("Failed to create Kafka producer: %s", e)
            return None

    def start_kafka_consumer(self):
        """
        Starts the Kafka consumer to listen for and process transactions in real-time.
        """
        if not self.consumer:
            logging.error("Kafka consumer is not available. Cannot start processing.")
            return

        logging.info("Starting real-time fraud detection consumer...")
        for message in self.consumer:
            transaction = message.value
            logging.info("Received transaction: %s", transaction['id'])
            
            # Preprocess the transaction data
            features = np.array(list(transaction['features'].values())).reshape(1, -1)
            scaled_features = self.scaler.transform(features)
            
            # Predict fraud
            prediction = self.model.predict(scaled_features)
            
            if prediction[0] == 1:
                self._send_fraud_alert(transaction)
    
    def _send_fraud_alert(self, transaction: dict):
        """
        Sends a fraud alert to the specified Kafka topic.
        
        Args:
            transaction (dict): The fraudulent transaction details.
        """
        if not self.producer:
            logging.error("Kafka producer is not available. Cannot send alert.")
            return
            
        alert = {
            'transaction_id': transaction['id'],
            'alert_message': 'High probability of fraudulent transaction detected!',
            'timestamp': time.time()
        }
        self.producer.send(self.kafka_config['fraud_alert_topic'], alert)
        self.producer.flush()
        logging.warning("Fraud alert sent for transaction %s", transaction['id'])

if __name__ == '__main__':
    kafka_settings = {
        'bootstrap_servers': ['localhost:9092'],
        'transaction_topic': 'credit_card_transactions',
        'fraud_alert_topic': 'fraud_alerts'
    }
    
    # Path to your best-performing trained model
    model_file_path = 'models/ensemble/stacking_fraud_detector.joblib'

    streaming_detector = RealTimeFraudDetector(model_path=model_file_path, kafka_config=kafka_settings)
    streaming_detector.start_kafka_consumer()
