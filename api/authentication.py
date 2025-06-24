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
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework import generics, status


# @method_decorator(csrf_exempt, name='dispatch')
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
        role = data.get('role', 'operator')  # Default to 'user'

        if role not in dict(UserProfile.ROLE_CHOICES):
            raise serializers.ValidationError({"error": f"Invalid role: {role}"})

        # Check if email already exists
        if UserProfile.objects.filter(email=email).exists():
            raise serializers.ValidationError(
                {"error": "Email already exists. Please choose a different email."}
            )

        # Create user with custom logic
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
                raise serializers.ValidationError(
                    {"error": "Username already exists. Please choose a different username."}
                )
            if 'user_profile.email' in str(e):
                raise serializers.ValidationError(
                    {"error": "Email already exists. Please choose a different email."}
                )
            raise serializers.ValidationError(
                {"error": "An error occurred during registration."}
            )

# @method_decorator(csrf_exempt, name='dispatch')
class LoginViewAPI(generics.CreateAPIView):
    serializer_class = LoginSerializer
    # permission_classes = [AllowAny]  # Allow anyone to access this view

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid(raise_exception=True):
            username = serializer.validated_data['username']
            password = serializer.validated_data['password']
            # Authenticate the user
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                try:
                    token_instance, token = AuthToken.objects.create(user)
                except Exception as e:
                    print("Error creating token:", e)
                    return Response({'message': "Failed to create token"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                # Retrieve user profile
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

                return Response(
                    {
                        'message': "Login Successful",
                        "data": profile_info,
                        'token': token,  # Return the token
                    },
                    status=status.HTTP_200_OK,
                )
            else:
                return Response({'message': "Invalid username or password"}, status=status.HTTP_401_UNAUTHORIZED)
            
@method_decorator(csrf_exempt, name='dispatch')
class LogoutViewAPI(APIView):
    
    # permission_classes = [IsAuthenticated]  # Only authenticated users can log out

    def post(self, request,token):
        print('aa')
        # Retrieve the user's token instance from the request
        try:
            auth_token_instance = KnoxAuthtoken.objects.get(token_key=token[:8])
        except :
            return Response({"message": "Invalid Token"}, status=status.HTTP_400_BAD_REQUEST)

        # auth_token_instance = request.auth  # Knox sets the AuthToken object in request.auth
        if auth_token_instance:
            try:
                # Delete the token from the database
                auth_token_instance.delete()
                return Response({"message": "Logout successful"}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"message": f"Error during logout: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response({"message": "Invalid token or already logged out"}, status=status.HTTP_400_BAD_REQUEST)


def token_verification(token):
    try:
        user = KnoxAuthtoken.objects.get(token_key= token).user
    except:
        return {'status':400,'error':'Invalid Token'}
    if user:
        return {'status':200,'user':user}
    else:
        return {'status':400,'error':'user not Found'}
