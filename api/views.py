import os
import uuid
import boto3
from rest_framework.views import APIView
from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework import status
from .models import AudioFile, SOP, SOPStep, Session
from .serializers import AudioFileSerializer, FeedbackSerializer, ProcessAudioViewSerializer, SOPSerializer, SessionSerializer
from .utils import *
from peercheck import settings
from fuzzywuzzy import fuzz
from Levenshtein import distance
from .authentication import token_verification
from .permissions import RoleBasedPermission
import logging

logger = logging.getLogger(__name__)

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
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {str(e)}")
    pass

MODEL_PATH = os.path.join(settings.BASE_DIR, "vosk-model-small-en-us-0.15")

SPEAKER_MODEL_PATH = os.path.join(settings.BASE_DIR, "vosk-model-spk-0.4")

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
    permission_classes = [RoleBasedPermission]

    def post(self, request, token,format=None):
        # Validate token
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)
        
        serializer = ProcessAudioViewSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        audio_file = serializer.validated_data.get("file")
        # start_prompt = request.data.get("start_prompt")
        # end_prompt = request.data.get("end_prompt")
        keywords = request.data.get("keywords", "")
        sop_id = request.data.get("sop_id") 
        session_id = serializer.validated_data.get("session_id")

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


        # Transcription with speaker diarization
        try:
            extracted_text = transcribe_with_speaker_diarization(file_url, MODEL_PATH, SPEAKER_MODEL_PATH)
            transcription_text = " ".join([segment["text"] for segment in extracted_text])
        except Exception as e:
            return Response({"error": f"Transcription failed: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Save to Database
        audio_instance = AudioFile.objects.create(
            file_path=file_url,
            transcription=transcription_text,
            status="processed",
            duration=len(transcription_text.split()),  # Word count as duration
            sop=None  # Will be updated if sop_id is provided
        )

        response_data = {
            "transcription": extracted_text,
            "status": "processed",
            "audio_file": AudioFileSerializer(audio_instance).data
        }

        

        # SOP Step Matching
        if sop_id:
            try:
                sop = SOP.objects.get(id=sop_id)
                sop_matches = match_sop_steps(transcription_text, sop)
                response_data["sop_matches"] = sop_matches

                # Update AudioFile with SOP
                audio_instance.sop = sop
                audio_instance.save()
            except SOP.DoesNotExist:
                response_data["sop_error"] = "SOP not found"
            except Exception as e:
                response_data["sop_error"] = f"SOP matching failed: {str(e)}"
        
        if session_id:
            try:
                session = Session.objects.get(id=session_id, user=user_data['user'])
                session.audio_files.add(audio_instance)
                logger.info(f"Audio file {audio_instance.id} added to session {session_id}")
            except Session.DoesNotExist:
                logger.error(f"Session not found or unauthorized: ID {session_id}")
                response_data["session_error"] = "Session not found or unauthorized"
            except Exception as e:
                logger.error(f"Session linking failed: {str(e)}")
                response_data["session_error"] = f"Session linking failed: {str(e)}"

        return Response(response_data, status=status.HTTP_200_OK)
    
        # # Transcription
        # try:
        #     # local_file_path = download_file_from_s3(file_name, S3_BUCKET_NAME)
        #     # print(local_file_path)
        #     transcription = process_audio_pipeline(file_url, MODEL_PATH)
        #     # os.remove(local_file_path)  # Clean up local file after processing
        # except Exception as e:
        #     return Response({"error": f"Transcription failed: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        # # Find prompts and extract text
        # start_index = find_prompt_index(transcription.lower(), start_prompt.lower())
        # end_index = find_prompt_index(transcription.lower(), end_prompt.lower())

        # # if start_index == -1 or end_index == -1 or start_index >= end_index:
        # #     return Response(
        # #         {"error": "Start or End Prompt not found in the audio."},
        # #         status=status.HTTP_400_BAD_REQUEST,
        # #     )

        # extracted_text = transcription[start_index:end_index]

        # # Detect Keywords
        # keyword_list = [k.strip() for k in keywords.split(",")]
        # detected_keywords = detect_keywords(transcription, keyword_list)

        # # Save to Database
        # audio_instance = AudioFile.objects.create(
        #     file_path=file_url,
        #     transcription=transcription,
        #     keywords_detected=detected_keywords,
        #     status="processed",
        #     duration=len(transcription.split()),  # Example: word count as duration
        # )

        # # Return Response
        # serializer = AudioFileSerializer(audio_instance)
        # return Response(
        #     {
        #         # "audio_file": serializer.data,
        #         "transcription": extracted_text,
        #         # "detected_keywords": detected_keywords,
        #         "status": "processed",
        #     },
        #     status=status.HTTP_200_OK,
        # )


class FeedbackView(APIView):
    permission_classes = [RoleBasedPermission]
    def post(self, request, format=None):
        serializer = FeedbackSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "Feedback submitted successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class GetAudioRecordsView(APIView):
    permission_classes = [RoleBasedPermission]

    def get(self, request, token, format=None):
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)
        try:
            # audio_records = AudioFile.objects.all().order_by("-id")
            if user_data['user'].role == 'admin':
                audio_records = AudioFile.objects.all()
            else:
                audio_records = AudioFile.objects.filter(sessions__user=user_data['user'])
            audio_records = audio_records.order_by("-id")
            serializer = AudioFileSerializer(audio_records, many=True)
            return Response({"audio_records": serializer.data}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"Error fetching audio records: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ReAnalyzeAudioView(APIView):
    permission_classes = [RoleBasedPermission]
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



class SOPCreateView(CreateAPIView):
    serializer_class = SOPSerializer
    permission_classes = [RoleBasedPermission]
    def post(self, request, token):
        # Validate token
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            serializer.save(created_by=user_data['user'])
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SOPListView(APIView):
    permission_classes = [RoleBasedPermission]
    def get(self, request, token):
        # Validate token
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)

        sops = SOP.objects.all()
        serializer = SOPSerializer(sops, many=True)
        return Response({'sops': serializer.data}, status=status.HTTP_200_OK)
    
class SessionCreateView(CreateAPIView):
    serializer_class = SessionSerializer
    permission_classes = [RoleBasedPermission]

    def post(self, request, token, format=None):
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            serializer.save(user=user_data['user'])
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SessionListView(APIView):
    permission_classes = [RoleBasedPermission]

    def get(self, request, token, format=None):
        user_data = token_verification(token)
        if user_data['status'] != 200:
            return Response({'error': user_data['error']}, status=status.HTTP_400_BAD_REQUEST)

        try:
            if user_data['user'].role == 'admin':
                sessions = Session.objects.all()
            else:
                sessions = Session.objects.filter(user=user_data['user'])
            sessions = sessions.order_by("-created_at")
            serializer = SessionSerializer(sessions, many=True)
            return Response({'sessions': serializer.data}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching sessions: {str(e)}")
            return Response({"error": f"Error fetching sessions: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

def transcribe_with_speaker_diarization(audio_url: str, model_path: str, speaker_model_path: str):
    """
    Transcribes audio with Vosk and includes speaker diarization.

    Args:
        audio_url (str): URL of the audio file.
        model_path (str): Path to the Vosk ASR model.
        speaker_model_path (str): Path to the Vosk speaker diarization model.

    Returns:
        dict: Formatted transcription with speaker labels.
    """
    # Download the file
    response = requests.get(audio_url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download file: {response.status_code}")
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(response.content)
        temp_audio_path = temp_audio.name

    try:
        model = vosk.Model(model_path)
        spk_model = vosk.SpkModel(speaker_model_path)  # Load Speaker Model

        with wave.open(temp_audio_path, "rb") as wf:
            recognizer = vosk.KaldiRecognizer(model, wf.getframerate())
            recognizer.SetWords(True)
            recognizer.SetSpkModel(spk_model)  # Enable speaker diarization

            transcription = []

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())

                    # Extract speaker ID
                    speaker_id = None
                    if "spk" in result:
                        if isinstance(result["spk"], list) and result["spk"]:  # Ensure it's a non-empty list
                            speaker_id = result["spk"][0]  # Take the first speaker ID
                        elif isinstance(result["spk"], (int, float)):  # Handle single value
                            speaker_id = result["spk"]

                    # Ensure valid speaker ID
                    speaker = f"Speaker_{int(speaker_id) + 1}" if speaker_id is not None else "Unknown"

                    # Append transcription
                    if "text" in result and result["text"]:
                        transcription.append({"speaker": speaker, "text": result["text"]})

            # Process the final result
            final_result = json.loads(recognizer.FinalResult())
            if "text" in final_result and final_result["text"]:
                speaker_id = None
                if "spk" in final_result:
                    if isinstance(final_result["spk"], list) and final_result["spk"]:
                        speaker_id = final_result["spk"][0]
                    elif isinstance(final_result["spk"], (int, float)):
                        speaker_id = final_result["spk"]

                speaker = f"Speaker_{int(speaker_id) + 1}" if speaker_id is not None else "Unknown"
                transcription.append({"speaker": speaker, "text": final_result["text"]})

    finally:
        os.remove(temp_audio_path)  # Delete temp file after processing

    return transcription
