import os
import uuid
import boto3
from rest_framework.views import APIView
from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework import status
from .models import AudioFile
from .serializers import AudioFileSerializer, FeedbackSerializer, ProcessAudioViewSerializer
from .utils import *
from peercheck import settings
from fuzzywuzzy import fuzz
from Levenshtein import distance
from .authentication import token_verification

try:
    # AWS S3 Configuration
    S3_BUCKET_NAME = settings.AWS_STORAGE_BUCKET_NAME
    S3_REGION = settings.AWS_S3_REGION_NAME
    S3_ACCESS_KEY = settings.AWS_S3_ACCESS_KEY_ID
    S3_SECRET_KEY = settings.AWS_S3_SECRET_ACCESS_KEY

    # Initialize S3 client
    s3_client = boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )
except :
    pass

MODEL_PATH = os.path.join(settings.BASE_DIR, "vosk-model-small-en-us-0.15")


def upload_file_to_s3(file, bucket_name, file_name):
    """
    Upload a file to an S3 bucket.
    """
    try:
        s3_client.upload_fileobj(file, bucket_name, file_name)
        return f"https://{bucket_name}.s3.amazonaws.com/{file_name}"
    except Exception as e:
        raise Exception(f"Error uploading file to S3: {str(e)}")


def download_file_from_s3(file_name, bucket_name):
    """
    Download a file from an S3 bucket.
    """
    try:
        file_path = f"/uploads/{uuid.uuid4()}_{os.path.basename(file_name)}"
        s3_client.download_file(bucket_name, file_name, file_path)
        return file_path
    except Exception as e:
        raise Exception(f"Error downloading file from S3: {str(e)}")


class ProcessAudioView(CreateAPIView):
    serializer_class = ProcessAudioViewSerializer

    def post(self, request, format=None):
        serializer = ProcessAudioViewSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        audio_file = serializer.validated_data.get("file")
        start_prompt = request.data.get("start_prompt")
        end_prompt = request.data.get("end_prompt")
        keywords = request.data.get("keywords", "")

        # if not audio_file or not start_prompt or not end_prompt:
        #     return Response({"error": "Missing Start or End Prompt fields."}, status=status.HTTP_400_BAD_REQUEST)

        # Generate unique file name for S3
        file_name = f"peercheck_files/{uuid.uuid4()}_{audio_file.name}"

        print(file_name)

        # Upload file to S3
        try:
            file_url = upload_file_to_s3(audio_file, S3_BUCKET_NAME, file_name)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        print(file_url)

        # Transcription
        try:
            # local_file_path = download_file_from_s3(file_name, S3_BUCKET_NAME)
            # print(local_file_path)
            transcription = process_audio_pipeline(file_url, MODEL_PATH)
            # os.remove(local_file_path)  # Clean up local file after processing
        except Exception as e:
            return Response({"error": f"Transcription failed: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        # Find prompts and extract text
        start_index = find_prompt_index(transcription.lower(), start_prompt.lower())
        end_index = find_prompt_index(transcription.lower(), end_prompt.lower())

        # if start_index == -1 or end_index == -1 or start_index >= end_index:
        #     return Response(
        #         {"error": "Start or End Prompt not found in the audio."},
        #         status=status.HTTP_400_BAD_REQUEST,
        #     )

        extracted_text = transcription[start_index:end_index]

        # Detect Keywords
        keyword_list = [k.strip() for k in keywords.split(",")]
        detected_keywords = detect_keywords(transcription, keyword_list)

        # Save to Database
        audio_instance = AudioFile.objects.create(
            file_path=file_url,
            transcription=transcription,
            keywords_detected=detected_keywords,
            status="processed",
            duration=len(transcription.split()),  # Example: word count as duration
        )

        # Return Response
        serializer = AudioFileSerializer(audio_instance)
        return Response(
            {
                "audio_file": serializer.data,
                "transcription": extracted_text,
                "detected_keywords": detected_keywords,
                "status": "processed",
            },
            status=status.HTTP_200_OK,
        )


class FeedbackView(APIView):
    def post(self, request, format=None):
        serializer = FeedbackSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "Feedback submitted successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class GetAudioRecordsView(APIView):
    """
    API to fetch all existing audio records.
    """

    def get(self, request, token,format=None):
        if not token:
            return Response({"error": "Token is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Validate user from token
        user = token_verification(token)
        if user['status'] != 200:
            return Response({'message': user['error']}, status=status.HTTP_400_BAD_REQUEST)
        try:
            audio_records = AudioFile.objects.all().order_by("-id")
            serializer = AudioFileSerializer(audio_records, many=True)
            return Response({"audio_records": serializer.data}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"Error fetching audio records: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ReAnalyzeAudioView(APIView):
    def post(self, request):
        file_path = request.data.get("file_path")
        new_keywords = request.data.get("new_keywords", "")
        # id = request.data.get("id")

        if not file_path or not new_keywords:
            return Response({"error": "File path and keywords are required"}, status=status.HTTP_400_BAD_REQUEST)

        # Download file from S3
        # try:
        #     local_file_path = download_file_from_s3(file_path, S3_BUCKET_NAME)
        # except Exception as e:
        #     return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)

        # Process transcription
        transcription = process_audio_pipeline(file_path, MODEL_PATH)
        # os.remove(local_file_path)  # Clean up local file after processing

        keywords_list = [kw.strip() for kw in new_keywords.split(",")]
        detected_keywords = detect_keywords(transcription, keywords_list)

        AudioFile.objects.filter(file_path=file_path).update(
            transcription=transcription,
            keywords_detected=detected_keywords,
            status="reanalyzed",
            duration=len(transcription.split()),  # Example: word count as duration
        )
        return Response(
            {
                "file_path": file_path,
                "transcription": transcription,
                "detected_keywords": detected_keywords,
            },
            status=status.HTTP_200_OK,
        )

def find_prompt_index(transcription, prompt, threshold=80):
    """
    Find the position of a prompt in transcription with fuzzy matching.
    
    :param transcription: Transcribed text from audio.
    :param prompt: The start or end prompt provided by the user.
    :param threshold: Matching threshold percentage.
    :return: Index of the best match if found, else -1.
    """
    words = transcription.lower().split()
    prompt_words = prompt.lower().split()
    best_match_index = -1

    for i in range(len(words) - len(prompt_words) + 1):
        segment = " ".join(words[i:i+len(prompt_words)])
        similarity = fuzz.ratio(segment, prompt.lower())

        if similarity >= threshold:
            best_match_index = transcription.lower().find(segment)
            break

    return best_match_index

def find_approximate_match(transcription, prompt):
    words = transcription.lower().split()
    prompt_words = prompt.lower().split()
    min_distance = float('inf')
    best_match_index = -1

    for i in range(len(words) - len(prompt_words) + 1):
        segment = " ".join(words[i:i+len(prompt_words)])
        current_distance = distance(segment, prompt.lower())

        if current_distance < min_distance:
            min_distance = current_distance
            best_match_index = transcription.lower().find(segment)

    return best_match_index if min_distance < len(prompt) * 0.3 else -1
