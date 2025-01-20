import os
from rest_framework.views import APIView
from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework import status
from .models import AudioFile
from .serializers import AudioFileSerializer, FeedbackSerializer, ProcessAudioViewSerializer
from .utils import detect_keywords, segment_transcription,process_audio_pipeline
import uuid
import logging
from fuzzywuzzy import fuzz, process
from Levenshtein import distance

UPLOAD_DIR = "./uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ProcessAudioView(CreateAPIView):
    serializer_class = ProcessAudioViewSerializer

    def post(self, request, format=None):
        serializer = ProcessAudioViewSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        audio_file = serializer.validated_data.get('file')
        start_prompt = request.data.get('start_prompt')
        end_prompt = request.data.get('end_prompt')
        keywords = request.data.get('keywords', '')  # Comma-separated keywords

        # audio_file = request.FILES.get('file')
        if not audio_file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        # Save the file
        file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{audio_file.name}")
        with open(file_path, "wb") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)


        # Transcription
        try:
            model_path = "vosk-model-en-us-0.22"
            transcription = process_audio_pipeline(file_path, model_path)
        except Exception as e:
            if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            return Response({"error": f"Transcription failed: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        # Analyze transcription for prompts
        # start_index = transcription.lower().find(start_prompt.lower())
        # end_index = transcription.lower().find(end_prompt.lower())

        # find_prompt_index
        start_index = find_prompt_index(transcription.lower(), start_prompt.lower())
        end_index = find_prompt_index(transcription.lower(), end_prompt.lower())

        # find_approximate_match
        # start_index = find_approximate_match(transcription, start_prompt)
        # end_index = find_approximate_match(transcription, end_prompt)
        if start_index == -1 or end_index == -1 or start_index >= end_index:
            os.remove(file_path)
            return Response(
                {"error": "Start or End Prompt not found in the audio."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Extract text between prompts
        extracted_text = transcription[start_index:end_index]

        # Detect keywords in the transcription
        # keyword_list = [k.strip() for k in keywords.split(',')]
        # print(keyword_list)
        
        # Keyword Detection
        # keywords = ["keyword1", "keyword2", "keyword3"]  # Replace with your keywords
        # keywords = [
        #     "AI", "Machine Learning", "Neural Networks", "Cloud Computing", "Blockchain",
        #     "Robotics", "Crypto", "Healthcare", "Mental Health", "Stocks",
        #     "Doctors", "Students", "Graduation", "Movies", "Music",
        #     "Football", "Tennis", "Sustainability", "Climate Change", "Elections",
        #     "Education", "Finance", "Investment", "Bonds", "Banking",
        #     "Marketing", "Sales", "Networking", "Leadership", "Consulting",
        #     "Entertainment", "Gaming", "Podcasts", "Olympics", "Cricket",
        #     "Rugby", "Baseball", "Renewable Energy", "Recycling", "Pollution",
        #     "Cybersecurity", "Data Privacy", "Innovation", "Artificial Intelligence", 
        #     "Data Science", "Deep Learning", "Natural Language Processing", "IoT", 
        #     "Agriculture", "Automation", "Smart Cities", "Smart Homes", "Augmented Reality",
        #     "Virtual Reality", "Cloud Storage", "Big Data", "Business Intelligence", "Digital Transformation"
        # ]

        detected_keywords = detect_keywords(transcription, keywords)

        # Segmentation
        # segments = segment_transcription(transcription)

        # Save to Database
        audio_instance = AudioFile.objects.create(
            file_path=file_path,
            transcription=transcription,
            keywords_detected=", ".join(detected_keywords),
            status="processed",
            duration=len(transcription.split())  # Example: word count as duration
        )
        # Return Response
        serializer = AudioFileSerializer(audio_instance)
        return Response({
            "audio_file": serializer.data,
            # "segments": segments,
            "extracted_text": extracted_text,
            "detected_keywords": detected_keywords,
        }, status=status.HTTP_200_OK)
    

# Feedback Submission View

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

    def get(self, request, format=None):
        try:
            # Fetch all records from the AudioFile model
            audio_records = AudioFile.objects.all().order_by('-id')
            serializer = AudioFileSerializer(audio_records, many=True)
            return Response({"audio_records": serializer.data}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"Error fetching audio records: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
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