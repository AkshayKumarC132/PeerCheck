from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import authenticate, login, logout
from django.views.decorators.csrf import csrf_exempt
from .serializers import *
from .models import *
from django.db import IntegrityError, transaction
from rest_framework.views import APIView
from knox.models import AuthToken
from django.utils.decorators import method_decorator
from rest_framework import generics, status

from .rag_integration import ensure_rag_token  # NEW

# ---------------- Register ----------------

class RegisterView(generics.CreateAPIView):
    serializer_class = UserProfileSerializer

    def perform_create(self, serializer):
        data = self.request.data
        required_fields = ['username', 'email', 'password']
        missing_or_empty_fields = [
            field for field in required_fields
            if field not in data or not data[field].strip()
        ]

        if missing_or_empty_fields:
            raise serializers.ValidationError(
                {"error": f"Fields cannot be null or empty: {', '.join(missing_or_empty_fields)}"}
            )

        username = data['username']
        email = data['email']
        password = data['password']
        role = data.get('role', 'operator')

        if role not in dict(UserProfile.ROLE_CHOICES):
            raise serializers.ValidationError({"error": f"Invalid role: {role}"})

        if UserProfile.objects.filter(email=email).exists():
            raise serializers.ValidationError({"error": "Email already exists. Please choose a different email."})

        try:
            user = UserProfile.objects.create_user(
                username=username,
                email=email,
                password=password,
                name=username,
                role=role
            )
            serializer.instance = user
        except IntegrityError as e:
            if 'user_profile.username' in str(e):
                raise serializers.ValidationError({"error": "Username already exists. Please choose a different username."})
            if 'user_profile.email' in str(e):
                raise serializers.ValidationError({"error": "Email already exists. Please choose a different email."})
            raise serializers.ValidationError({"error": "An error occurred during registration."})

# ---------------- Login ----------------

class LoginViewAPI(generics.CreateAPIView):
    serializer_class = LoginSerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid(raise_exception=True):
            username = serializer.validated_data['username']
            password = serializer.validated_data['password']
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                try:
                    token_instance, token = AuthToken.objects.create(user)
                except Exception:
                    return Response({'message': "Failed to create token"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                profile = UserProfile.objects.get(username=username)
                profile_info = {
                    "last_login": profile.last_login,
                    "username": profile.username,
                    "first_name": profile.first_name,
                    "last_name": profile.last_name,
                    "email": profile.email,
                    "date_joined": profile.date_joined,
                    "id": profile.id,
                    "name": profile.name,
                    "created_at": profile.created_at,
                    "updated_at": profile.updated_at,
                    "theme": profile.theme,
                    "role": profile.role
                }

                # NEW: ensure RAG token (non-blocking if disabled)
                ensure_rag_token(profile)

                return Response({
                        'message': "Login Successful",
                        "data": profile_info,
                        'token': token,
                    }, status=status.HTTP_200_OK)
            else:
                return Response({'message': "Invalid username or password"}, status=status.HTTP_401_UNAUTHORIZED)

# ---------------- Logout ----------------
@method_decorator(csrf_exempt, name='dispatch')
class LogoutViewAPI(APIView):
    def post(self, request, token):
        try:
            auth_token_instance = KnoxAuthtoken.objects.get(token_key=token[:8])
        except :
            return Response({"message": "Invalid Token"}, status=status.HTTP_400_BAD_REQUEST)

        if auth_token_instance:
            try:
                auth_token_instance.delete()
                # NOTE: Do NOT clear RAG token here (per requirements)
                return Response({"message": "Logout successful"}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"message": f"Error during logout: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response({"message": "Invalid token or already logged out"}, status=status.HTTP_400_BAD_REQUEST)

# ---------------- Token check helper (used by views) ----------------

def token_verification(token_key: str):
    """
    Verifies a 3PC (Knox) token and returns a standardized dict.
    Does not touch or clear any RAG tokens.
    """
    try:
        token = AuthToken.objects.get(token_key=token_key)
    except AuthToken.DoesNotExist:
        return {"status": 401, "error": "Invalid or expired token."}

    try:
        user: UserProfile = token.user
        if not user or not user.is_active:
            return {"status": 401, "error": "User inactive or not found."}
        return {"status": 200, "user": user}
    except Exception:
        return {"status": 401, "error": "Authentication failed."}
