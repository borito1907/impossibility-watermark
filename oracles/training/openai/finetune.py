# RUN: python -m oracles.training.openai.finetune

import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import time

class FineTuneHelper:
    def __init__(self):
        # Load environment variables from a .env file
        load_dotenv()

        # Get the OpenAI API key from the environment
        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
        
        # Initialize the OpenAI client with the API key
        self.client = OpenAI(api_key=api_key)

    def upload_training_file(self, training_file_path):
        """Upload a training file to the OpenAI API for fine-tuning."""
        with open(training_file_path, "rb") as f:
            response = self.client.files.create(
                file=f,
                purpose="fine-tune"
            )
        
        # Depending on the OpenAI SDK version, response might need to be accessed differently
        try:
            training_file_id = response['id']  # If the response is a dictionary
        except TypeError:
            # If 'response' is an object and doesn't support indexing
            training_file_id = response.id  # Access the 'id' attribute directly

        print(f"Dataset uploaded successfully with file ID: {training_file_id}")
        return training_file_id

    def create_fine_tuning_job(self, training_file_id, model_name='gpt-4o-mini', suffix=None):
        """Create a fine-tuning job using the uploaded training file."""
        fine_tune_job = self.client.fine_tuning.jobs.create(
            training_file=training_file_id, 
            model=model_name,
            suffix=suffix
        )
        
        # Access the 'id' attribute directly if 'fine_tune_job' is an object
        try:
            fine_tune_job_id = fine_tune_job['id']  # If it's a dictionary
        except TypeError:
            fine_tune_job_id = fine_tune_job.id  # If it's an object

        print(f"Fine-tuning started with Job ID: {fine_tune_job_id}")
        return fine_tune_job_id

    def retrieve_fine_tuning_job(self, job_id, check_interval=60):
        """Continuously retrieve the status of a fine-tuning job until it completes."""
        while True:
            job_status = self.client.fine_tuning.jobs.retrieve(job_id)

            # Access the 'status' attribute directly if 'job_status' is an object
            try:
                status = job_status['status']  # If it's a dictionary
            except TypeError:
                status = job_status.status  # If it's an object

            print(f"Fine-tuning job {job_id} status: {status}")

            if status in ['succeeded', 'failed']:
                print(f"Fine-tuning job {job_id} has completed with status: {status}")
                break

            print(f"Waiting for {check_interval} seconds before checking status again...")
            time.sleep(check_interval)
        
        return job_status


    def list_fine_tuning_jobs(self, limit=10):
        """List fine-tuning jobs."""
        jobs = self.client.fine_tuning.jobs.list(limit=limit)
        return jobs

    def cancel_fine_tuning_job(self, job_id):
        """Cancel a fine-tuning job."""
        cancel_response = self.client.fine_tuning.jobs.cancel(job_id)
        print(f"Fine-tuning job {job_id} has been canceled.")
        return cancel_response

    def list_fine_tuning_events(self, job_id, limit=10):
        """List events from a fine-tuning job."""
        events = self.client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=limit)
        return events

    def delete_fine_tuned_model(self, model_id):
        """Delete a fine-tuned model."""
        delete_response = self.client.models.delete(model_id)
        print(f"Model {model_id} has been deleted.")
        return delete_response


if __name__ == "__main__":

    helper = FineTuneHelper()

    train_path = "./oracles/training/openai/data/train.jsonl"
    training_file_id = helper.upload_training_file(train_path)
    fine_tune_job_id = helper.create_fine_tuning_job(training_file_id, model_name='gpt-4o-2024-08-06')
    status = helper.retrieve_fine_tuning_job(fine_tune_job_id)
