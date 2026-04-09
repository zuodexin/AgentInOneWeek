---
name: video_ingestion
description: A skill for importing and managing video files in the system, when the user mentioned a new video or a person not in the database, the agent can use this skill to import the video and extract the person's information into the database.
license: Apache-2.0
metadata:
  author: example-org
  version: "1.0"
---

# Step-by-step instructions
1. localize the video file by the video name or the person's name mentioned by the user.
2. localize the database schema and make sure the database in ${pwd}/database/data.db is ready for importing, create the database if necessary.
3. check if the video file is already in the database, if yes, skip the importing process and return the existing information. if the video file is not in the database, continue to the next step.
4. Identify your workspace folder for saving intermediate files, you can use ./workspace for this purpose. do not clean up the workspace folder, you can reuse the intermediate files in the workspace for future use.
5. use ffmpeg to extract the audio from the video file, and save it as a temporary audio file, do not consume too much time for audio extraction.
6. check if there is a ASR result, if not, use ASR tool to transcribe the audio file and get the transcript.
7. check if the ASR result exists in the database, if not, store it in the database, you need to write python script to store the ASR results in the database.
8. show the brief information of the database update.
9. report the whole process.

# Examples of inputs and outputs

input: 
- video_name, "the name of the video, i.e. example.mp4"

output:
- update_info, "the information that prove the database is updated"

# Common edge cases