import unittest
import sys
import os

# Ensure the flask_app directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from flask_app.app import app

class FraudDetectionAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page_get(self):
        """Test if home page loads correctly"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>', response.data)  # Ensure HTML title exists

    def test_home_page_post_valid(self):
        """Test if home page POST request with valid input works"""
        valid_input = ",".join(["1.0"] * 30)  # 30 valid float values
        response = self.client.post('/', data={"csv_input": valid_input})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'Fraud' in response.data or b'Non-Fraud' in response.data,
            "Response should contain either 'Fraud' or 'Non-Fraud'"
        )

    def test_home_page_post_invalid(self):
        """Test home page POST request with invalid input"""
        response = self.client.post('/', data={"csv_input": "invalid,data,here"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Input Error', response.data)  # Ensure error message

    def test_predict_valid(self):
        """Test prediction endpoint with valid input"""
        valid_input = ",".join(["1.0"] * 30)  # 30 valid float values
        response = self.client.post('/predict', data={"csv_input": valid_input})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Fraud', response.data) or self.assertIn(b'Non-Fraud', response.data)

    def test_predict_invalid(self):
        """Test prediction endpoint with invalid input"""
        response = self.client.post('/predict', data={"csv_input": "invalid,data,here"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Error processing input', response.data)  # Expect error message

    # def test_metrics_endpoint(self):
    #     """Test if metrics endpoint is accessible"""
    #     response = self.client.get('/metrics')
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn(b'app_request_count', response.data)  # Check for Prometheus metric

if __name__ == '__main__':
    unittest.main()

