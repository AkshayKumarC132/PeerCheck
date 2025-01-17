import os
from rest_framework.views import APIView
from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework import status
from .models import AudioFile
from .serializers import AudioFileSerializer, FeedbackSerializer, ProcessAudioViewSerializer
from .utils import detect_keywords, segment_transcription,process_audio_pipeline


UPLOAD_DIR = "./uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ProcessAudioView(CreateAPIView):
    serializer_class = ProcessAudioViewSerializer

    def post(self, request, format=None):
        serializer = ProcessAudioViewSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        audio_file = serializer.validated_data.get('file')
        print(audio_file)

        # audio_file = request.FILES.get('file')
        if not audio_file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        file_path = os.path.join(UPLOAD_DIR, audio_file.name)
        with open(file_path, "wb") as f:
            f.write(audio_file.read())


        # Transcription
        # try:
        #     model_path = "vosk-model-en-us-0.22"
        #     transcription = process_audio_pipeline_optimized(file_path, model_path)
        # except Exception as e:
        #     return Response({"error": f"Transcription failed: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        import logging

        logging.basicConfig(level=logging.INFO)

        logging.info("Starting transcription...")

        try:
            model_path = "vosk-model-en-us-0.22"
            # vosk-model-small-en-us-0.15
            transcription = process_audio_pipeline(file_path, model_path)
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            logging.basicConfig(filename="server.log", level=logging.INFO)
            return Response({"error": f"Transcription failed: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        # Keyword Detection
        # keywords = ["keyword1", "keyword2", "keyword3"]  # Replace with your keywords
        keywords = [
            "AI", "Machine Learning", "Neural Networks", "Cloud Computing", "Blockchain",
            "Robotics", "Crypto", "Healthcare", "Mental Health", "Stocks",
            "Doctors", "Students", "Graduation", "Movies", "Music",
            "Football", "Tennis", "Sustainability", "Climate Change", "Elections",
            "Education", "Finance", "Investment", "Bonds", "Banking",
            "Marketing", "Sales", "Networking", "Leadership", "Consulting",
            "Entertainment", "Gaming", "Podcasts", "Olympics", "Cricket",
            "Rugby", "Baseball", "Renewable Energy", "Recycling", "Pollution",
            "Cybersecurity", "Data Privacy", "Innovation", "Artificial Intelligence", 
            "Data Science", "Deep Learning", "Natural Language Processing", "IoT", 
            "Agriculture", "Automation", "Smart Cities", "Smart Homes", "Augmented Reality",
            "Virtual Reality", "Cloud Storage", "Big Data", "Business Intelligence", "Digital Transformation"
        ]

        detected_keywords = detect_keywords(transcription, keywords)

        # Segmentation
        segments = segment_transcription(transcription)

        # Save to Database
        audio_instance = AudioFile.objects.create(
            file_path=file_path,
            transcription=transcription,
            keywords_detected=", ".join(detected_keywords),
            status="processed",
            duration=len(transcription.split())  # Example: word count as duration
        )

        print("Audio Instance :",audio_instance)

        # Return Response
        serializer = AudioFileSerializer(audio_instance)
        print('-----------')
        print("Serializer Data :",serializer.data)
        return Response({
            "audio_file": serializer.data,
            "segments": segments
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