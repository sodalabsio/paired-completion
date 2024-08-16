from openai import OpenAI

client = OpenAI()
import time
from datetime import datetime


def upload_file(path):
    file = client.files.create(file=open(path, "r"),
    purpose="fine-tune")

    # Wait for file to be processed
    while True:
        file = client.files.retrieve(file.id)
        print(file.status)
        if file.status == "processed":
            break
        if file.status == "error":
            # Get the error content
            error = client.files.retrieve(file.id)
            print(error)
            raise Exception("File processing failed")
            
        time.sleep(6)

    return file

def fine_tune(training_file_id, model, suffix, validation_file_id=None, watch=True, **kwargs):
    job = client.fine_tuning.create(training_file=training_file_id,
    validation_file=validation_file_id,
    model=model,
    suffix=suffix,
    **kwargs)

    print("Started job with id:", job.id)
    if watch:
        watch_job(job)

    return job

def watch_job(job, job_id=None):
    if job_id is None:
        job_id = job.id
    cur_event_count = 0

    retrieved = client.fine_tuning.retrieve(job_id)
    print("*** Job", retrieved.id, "***")
    print("Status:", retrieved.status)
    print("Hyperparameters:")
    for k, v in retrieved.hyperparameters.items():
        print(f"  {k}: {v}")
    print()

    while True:
        events = client.fine_tuning.list_events(job_id)
        event_list = events.data[::-1]

        if len(event_list) > cur_event_count:
            for event in event_list[cur_event_count:]:
                event_type = event.type
                event_message = event.message
                event_time = event.created_at
                # Convert time like 1692932936 to local time
                event_time = datetime.fromtimestamp(event_time)
                print(f"{event_time}: [{event_type}] '{event_message}'")
            cur_event_count = len(event_list)
        
        retrieved = client.fine_tuning.retrieve(job_id)
        if retrieved.status == "succeeded":
            break
        if retrieved.status in ["cancelled", "failed", "stopped"]:
            raise Exception("Job failed with status '{}'".format(retrieved.status))
        time.sleep(10)

    return job