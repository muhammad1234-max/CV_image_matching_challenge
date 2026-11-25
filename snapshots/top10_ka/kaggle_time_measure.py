from kaggle.api.kaggle_api_extended import KaggleApi
import datetime
from datetime import timezone
import time

api = KaggleApi()
api.authenticate()

COMPETITION = "image-matching-challenge-2025"
result_ = api.competition_submissions(COMPETITION)[0]
latest_ref = result_.ref
submit_time = result_.date

status = ""

while str(status) != "SubmissionStatus.COMPLETE":
    list_of_submission = api.competition_submissions(COMPETITION)
    for result in list_of_submission:
        if result.ref == latest_ref:
            break
    status = result.status

    now = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
    elapsed_time = int((now - submit_time).seconds / 60) + 1
    if str(status) == "SubmissionStatus.COMPLETE":
        print("\r", f"run-time: {elapsed_time} min, LB: {result.public_score}")
    else:
        print("\r", f"elapsed time: {elapsed_time} min", end="")
        time.sleep(60)
