import boto3
import pandas as pd
import os
import sys
import logging
from src.logger import logging
from io import StringIO
from src.connections import s3_connection



class s3_operations:
    def __init__(self, bucket_name, aws_access_key, aws_secret_key, region_name="ap-southeast-2"):
        """
        Initialize the s3_operations class with AWS credentials and S3 bucket details.
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
        logging.info("Data Ingestion from S3 bucket initialized")

    def fetch_file_from_s3(self, file_key):
        """
        Fetches a CSV file from the S3 bucket and returns it as a Pandas DataFrame.
        :param file_key: S3 file path (e.g., 'data/data.csv')
        :return: Pandas DataFrame
        """
        try:
            logging.info(f"Fetching file '{file_key}' from S3 bucket '{self.bucket_name}'...")
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            logging.info(f"Successfully fetched and loaded '{file_key}' from S3 that has {len(df)} records.")
            return df
        except Exception as e:
            logging.exception(f"❌ Failed to fetch '{file_key}' from S3: {e}")
            return None

# # Example usage
# # if __name__ == "__main__":
#         bucket_name = "learnyardproj1-bucket"
#         aws_access_key = ""
#         aws_secret_key = ""
#         FILE_KEY = "creditcard.csv"  # Path inside S3 bucket
        
#         s3 = s3_connection.s3_opeations(bucket_name,aws_access_key,aws_secret_key)
#         df = s3.fetch_file_from_s3("FILE_KEY")
         

# #         print(f"Data fetched with {len(df)} records..")  # Display first few rows of the fetched DataFrame
        