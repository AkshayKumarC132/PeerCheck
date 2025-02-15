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


@api_view(['POST'])
@transaction.atomic()
def register(request):
    required_fields = ['username', 'email', 'password']

    # Check if required fields are present and not empty
    missing_or_empty_fields = [
        field for field in required_fields
        if field not in request.data or not request.data[field].strip()
    ]

    if missing_or_empty_fields:
        return Response(
            {"error": f"Fields cannot be null or empty: {', '.join(missing_or_empty_fields)}"},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Extracting fields
    username = request.data['username']
    email = request.data['email']
    password = request.data['password']
    # name = request.data['name']

    if UserProfile.objects.filter(email=email).exists():
        return Response(
            {"error": "Email already exists. Please choose a different email."},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        # Creating the user profile
        user = UserProfile.objects.create_user(
            username=username,
            email=email,
            password=password,
            name=username,
        )
        user.save()

    except IntegrityError as e:
        if 'user_profile.username' in str(e):
            return Response(
                {"error": "Username already exists. Please choose a different username."},
                status=status.HTTP_400_BAD_REQUEST
            )
        if 'user_profile.email' in str(e):
            return Response(
                {"error": "Email already exists. Please choose a different email."},
                status=status.HTTP_400_BAD_REQUEST
            )
        return Response(
            {"error": "An error occurred during registration."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    # Serialize and return the newly created user profile
    serializer = UserProfileSerializer(user)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


class LoginViewAPI(APIView):
    serializer_class = LoginSerialzier
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
                profile = UserProfile.objects.filter(username=username).values()

                return Response(
                    {
                        'message': "Login Successful",
                        "data": profile,
                        'token': token,  # Return the token
                    },
                    status=status.HTTP_200_OK,
                )
            else:
                return Response({'message': "Invalid username or password"}, status=status.HTTP_401_UNAUTHORIZED)
            

class LogoutViewAPI(APIView):
    
    # permission_classes = [IsAuthenticated]  # Only authenticated users can log out

    def post(self, request,token):
        print('aa')
        # Retrieve the user's token instance from the request
        try:
            auth_token_instance = KnoxAuthtoken.objects.get(token_key=token)
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