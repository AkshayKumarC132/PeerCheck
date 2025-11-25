import ast
import logging
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response
from rest_framework.views import APIView
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema

from .authentication import token_verification
from .models import (
    AudioFile,
    AuditLog,
    ReferenceDocument,
    SystemSettings,
    UserProfile,
    UserSettings,
)
from .permissions import RoleBasedPermission
from .serializers import (
    AdminUserProfileSerializer,
    AudioFileSerializer,
    AuditLogSerializer,
    ErrorResponseSerializer,
    SystemSettingsSerializer,
    UserProfileSerializer,
    UserSettingsSerializer,
)

logger = logging.getLogger(__name__)


class GetAudioRecordsView(APIView):
    """
    Retrieves a list of audio records based on user role and permissions.
    Supports pagination.
    """

    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="List Audio Records",
        description=(
            "Fetches a paginated list of audio records. Access is role-dependent: "
            "Admins see all, Operators see their own or those in their sessions, "
            "Reviewers see those in sessions they are part of."
        ),
        parameters=[
            OpenApiParameter(
                "token",
                OpenApiTypes.STR,
                OpenApiParameter.PATH,
                description="Authentication token.",
            ),
            OpenApiParameter(
                "page",
                OpenApiTypes.INT,
                OpenApiParameter.QUERY,
                description="Page number for pagination.",
            ),
        ],
        responses={
            200: AudioFileSerializer(many=True),
            401: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        },
        tags=["Audio Records"],
    )
    def get(self, request, token, format=None):
        """
        Lists audio records with pagination.
        - Admins: Can see all audio records.
        - Operators: Can see audio records they uploaded or those part of their sessions.
        - Reviewers: Can see audio records part of sessions they are assigned to.
        """
        user_data = token_verification(token)
        if user_data["status"] != 200:
            return Response(
                {"error": user_data["error"]}, status=status.HTTP_400_BAD_REQUEST
            )

        user = user_data["user"]
        try:
            audio_records = AudioFile.objects.filter(user=user).order_by("-created_at")

            paginator = PageNumberPagination()
            paginated_queryset = paginator.paginate_queryset(audio_records, request, view=self)
            serializer = AudioFileSerializer(paginated_queryset, many=True)
            response_data = serializer.data

            for item in response_data:
                keywords = item.get("keywords_detected", None)
                if isinstance(keywords, str):
                    try:
                        item["keywords_detected"] = ast.literal_eval(keywords)
                    except (ValueError, SyntaxError):
                        item["keywords_detected"] = []

            return paginator.get_paginated_response(response_data)
        except Exception as e:
            logger.error(f"Error fetching audio records with pagination: {str(e)}")
            return Response(
                {"error": f"Error fetching audio records: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class AdminDashboardSummaryView(APIView):
    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="Admin Dashboard Summary",
        description=(
            "Provides a summary of key metrics for the admin dashboard. "
            "Requires 'admin' role."
        ),
        parameters=[
            OpenApiParameter(
                "token",
                OpenApiTypes.STR,
                OpenApiParameter.PATH,
                description="Authentication token.",
            )
        ],
        responses={
            200: OpenApiTypes.OBJECT,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        },
        tags=["Admin"],
    )
    def get(self, request, token):
        if request.validated_user.role != "admin":
            return Response(
                {"error": "Forbidden. Admin access required."},
                status=status.HTTP_403_FORBIDDEN,
            )

        user_data = token_verification(token)
        if user_data["status"] != 200:
            return Response(
                {"error": user_data["error"]}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            total_users = UserProfile.objects.filter(id=user_data["user"].id).count()
            total_audio_files = AudioFile.objects.filter(user=user_data["user"].id).count()
            processed_audio_files = AudioFile.objects.filter(
                status="processed", user=user_data["user"].id
            ).count()
            failed_audio_files = AudioFile.objects.filter(
                status="failed", user=user_data["user"].id
            ).count()
            total_documents = ReferenceDocument.objects.filter(
                uploaded_by=user_data["user"].id
            ).count()

            data = {
                "total_users": total_users,
                "total_audio_files": total_audio_files,
                "total_documents": total_documents,
                "processed_audio_files": processed_audio_files,
                "failed_audio_files": failed_audio_files,
            }
            return Response(data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error generating admin dashboard summary: {str(e)}")
            return Response(
                {"error": "An error occurred while generating the summary."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class AdminUserListView(APIView):
    """
    Admin view to list all user profiles with pagination.
    """

    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="Admin: List Users",
        description="Retrieves a paginated list of all user profiles. Requires 'admin' role.",
        parameters=[
            OpenApiParameter(
                "token",
                OpenApiTypes.STR,
                OpenApiParameter.PATH,
                description="Authentication token.",
            ),
            OpenApiParameter(
                "page",
                OpenApiTypes.INT,
                OpenApiParameter.QUERY,
                description="Page number for pagination.",
            ),
        ],
        responses={
            200: AdminUserProfileSerializer(many=True),
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
        },
        tags=["Admin User Management"],
    )
    def get(self, request, token):
        if request.validated_user.role != "admin":
            return Response(
                {"error": "Forbidden. Admin access required."},
                status=status.HTTP_403_FORBIDDEN,
            )

        users = UserProfile.objects.all().order_by("id")

        paginator = PageNumberPagination()
        paginated_queryset = paginator.paginate_queryset(users, request, view=self)
        serializer = AdminUserProfileSerializer(paginated_queryset, many=True)
        return paginator.get_paginated_response(serializer.data)


class AdminUserDetailView(APIView):
    """
    Admin view to retrieve, update, and delete a specific user profile.
    """

    permission_classes = [RoleBasedPermission]

    def get_object(self, user_id):
        try:
            return UserProfile.objects.get(pk=user_id)
        except UserProfile.DoesNotExist:
            return None

    @extend_schema(
        summary="Admin: Retrieve User Details",
        description=(
            "Fetches the details of a specific user profile by their ID. "
            "Requires 'admin' role."
        ),
        parameters=[
            OpenApiParameter(
                "user_id",
                OpenApiTypes.INT,
                OpenApiParameter.PATH,
                description="The ID of the user to retrieve.",
            ),
            OpenApiParameter(
                "token",
                OpenApiTypes.STR,
                OpenApiParameter.PATH,
                description="Authentication token.",
            ),
        ],
        responses={
            200: AdminUserProfileSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=["Admin User Management"],
    )
    def get(self, request, user_id, token):
        if request.validated_user.role != "admin":
            return Response(
                {"error": "Forbidden. Admin access required."},
                status=status.HTTP_403_FORBIDDEN,
            )

        user_profile = self.get_object(user_id)
        if user_profile is None:
            return Response(
                {"error": "UserProfile not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = AdminUserProfileSerializer(user_profile)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @extend_schema(
        summary="Admin: Update User",
        description=(
            "Updates an existing user profile. Requires 'admin' role. "
            "Admins cannot change their own role or active status via this endpoint."
        ),
        parameters=[
            OpenApiParameter(
                "user_id",
                OpenApiTypes.INT,
                OpenApiParameter.PATH,
                description="The ID of the user to update.",
            ),
            OpenApiParameter(
                "token",
                OpenApiTypes.STR,
                OpenApiParameter.PATH,
                description="Authentication token.",
            ),
        ],
        request=AdminUserProfileSerializer,
        responses={
            200: AdminUserProfileSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=["Admin User Management"],
    )
    def put(self, request, user_id, token):
        return self._update_handler(request, user_id, token, partial=False)

    @extend_schema(
        summary="Admin: Partially Update User",
        description=(
            "Partially updates an existing user profile. Requires 'admin' role. "
            "Admins cannot change their own role or active status via this endpoint."
        ),
        parameters=[
            OpenApiParameter(
                "user_id",
                OpenApiTypes.INT,
                OpenApiParameter.PATH,
                description="The ID of the user to partially update.",
            ),
            OpenApiParameter(
                "token",
                OpenApiTypes.STR,
                OpenApiParameter.PATH,
                description="Authentication token.",
            ),
        ],
        request=AdminUserProfileSerializer,
        responses={
            200: AdminUserProfileSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=["Admin User Management"],
    )
    def patch(self, request, user_id, token):
        return self._update_handler(request, user_id, token, partial=True)

    def _update_handler(self, request, user_id, token, partial):
        if request.validated_user.role != "admin":
            return Response(
                {"error": "Forbidden. Admin access required."},
                status=status.HTTP_403_FORBIDDEN,
            )

        user_profile = self.get_object(user_id)
        if user_profile is None:
            return Response(
                {"error": "UserProfile not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if user_profile == request.validated_user and (
            "role" in request.data or "is_active" in request.data
        ):
            logger.warning(
                "Admin user %s attempted to change their own role or active status via API.",
                request.validated_user.username,
            )
            return Response(
                {"error": "Admins cannot change their own role or active status via this API."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer = AdminUserProfileSerializer(
            user_profile, data=request.data, partial=partial
        )
        if serializer.is_valid():
            updated_user_profile = serializer.save()
            AuditLog.objects.create(
                action="userprofile_update",
                user=request.validated_user,
                object_id=updated_user_profile.id,
                object_type="UserProfile",
                details={
                    "target_user_id": updated_user_profile.id,
                    "target_username": updated_user_profile.username,
                    "updated_fields": list(request.data.keys()),
                },
            )
            logger.info(
                "UserProfile %s (%s) updated by admin %s",
                updated_user_profile.id,
                updated_user_profile.username,
                request.validated_user.username,
            )
            return Response(
                AdminUserProfileSerializer(updated_user_profile).data,
                status=status.HTTP_200_OK,
            )
        logger.error(
            "Error updating UserProfile %s: %s", user_id, serializer.errors
        )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @extend_schema(
        summary="Admin: Delete User",
        description=(
            "Deletes an existing user profile. Requires 'admin' role. "
            "Admins cannot delete themselves via this endpoint."
        ),
        parameters=[
            OpenApiParameter(
                "user_id",
                OpenApiTypes.INT,
                OpenApiParameter.PATH,
                description="The ID of the user to delete.",
            ),
            OpenApiParameter(
                "token",
                OpenApiTypes.STR,
                OpenApiParameter.PATH,
                description="Authentication token.",
            ),
        ],
        responses={
            204: OpenApiTypes.NONE,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=["Admin User Management"],
    )
    def delete(self, request, user_id, token):
        if request.validated_user.role != "admin":
            return Response(
                {"error": "Forbidden. Admin access required."},
                status=status.HTTP_403_FORBIDDEN,
            )

        user_profile_to_delete = self.get_object(user_id)
        if user_profile_to_delete is None:
            return Response(
                {"error": "UserProfile not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if user_profile_to_delete == request.validated_user:
            logger.warning(
                "Admin user %s attempted to delete themselves via API.",
                request.validated_user.username,
            )
            return Response(
                {"error": "Admin users cannot delete themselves via the API."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        username_for_log = user_profile_to_delete.username
        user_profile_to_delete.delete()

        AuditLog.objects.create(
            action="userprofile_delete",
            user=request.validated_user,
            object_id=user_id,
            object_type="UserProfile",
            details={"deleted_user_id": user_id, "deleted_username": username_for_log},
        )
        logger.info(
            "UserProfile %s (%s) deleted by admin %s",
            user_id,
            username_for_log,
            request.validated_user.username,
        )
        return Response(status=status.HTTP_204_NO_CONTENT)


class DashboardSummaryView(APIView):
    def get(self, request, token):
        """
        Returns dashboard summary metrics for the authenticated user.
        """
        user_data = token_verification(token)
        if user_data["status"] != 200:
            return Response({"error": user_data["error"]}, status=400)

        total_documents = ReferenceDocument.objects.filter(
            uploaded_by=user_data["user"].id
        ).count()
        total_audio_files = AudioFile.objects.filter(user=user_data["user"].id).count()
        processed_audio = AudioFile.objects.filter(
            status="processed",
            diarization__isnull=False,
            user=user_data["user"].id,
        ).count()
        pending_diarization = AudioFile.objects.filter(
            status="processed",
            diarization__isnull=True,
            user=user_data["user"].id,
        ).count()
        recent_activity = AuditLog.objects.filter(user=user_data["user"]).order_by(
            "-timestamp"
        )[:5]

        data = {
            "total_documents": total_documents,
            "total_audio_files": total_audio_files,
            "processed_audio": processed_audio,
            "pending_diarization": pending_diarization,
            "recent_activity": AuditLogSerializer(recent_activity, many=True).data,
        }
        return Response(data)


class UserSettingsView(APIView):
    """
    Manages user-specific settings.
    """

    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="Get User Settings",
        description="Retrieves the settings for the authenticated user.",
        parameters=[
            OpenApiParameter(
                "token",
                OpenApiTypes.STR,
                OpenApiParameter.PATH,
                description="Authentication token.",
            )
        ],
        responses={
            200: UserSettingsSerializer,
            401: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        },
        tags=["User Settings"],
    )
    def get(self, request, token, format=None):
        logger.info("Fetching user settings for user: %s", request.validated_user.username)
        try:
            settings_obj, _ = UserSettings.objects.get_or_create(user=request.validated_user)
            serializer = UserSettingsSerializer(settings_obj)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching user settings: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @extend_schema(
        summary="Update User Settings",
        description=(
            "Partially updates settings for the authenticated user. "
            "Allows changing language, notification preferences, and theme."
        ),
        parameters=[
            OpenApiParameter(
                "token",
                OpenApiTypes.STR,
                OpenApiParameter.PATH,
                description="Authentication token.",
            )
        ],
        request=UserSettingsSerializer,
        responses={
            200: UserSettingsSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        },
        tags=["User Settings"],
    )
    def patch(self, request, token, format=None):
        logger.info("Updating user settings for user: %s", request.validated_user.username)
        try:
            settings_obj, _ = UserSettings.objects.get_or_create(user=request.validated_user)
            serializer = UserSettingsSerializer(settings_obj, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                if "theme" in request.data:
                    request.validated_user.theme = request.data["theme"]
                    request.validated_user.save()
                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error updating user settings: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SystemSettingsView(APIView):
    """
    Manages system-wide settings.
    Accessible only by Admin users.
    """

    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="Get System Settings",
        description="Retrieves system-wide settings. Requires 'admin' role.",
        parameters=[
            OpenApiParameter(
                "token",
                OpenApiTypes.STR,
                OpenApiParameter.PATH,
                description="Authentication token.",
            )
        ],
        responses={
            200: SystemSettingsSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        },
        tags=["System Settings"],
    )
    def get(self, request, token, format=None):
        logger.info("Fetching system settings for user: %s", request.validated_user.username)
        try:
            settings_obj, _ = SystemSettings.objects.get_or_create(id=1)
            serializer = SystemSettingsSerializer(settings_obj)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching system settings: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @extend_schema(
        summary="Update System Settings",
        description="Partially updates system-wide settings. Requires 'admin' role.",
        parameters=[
            OpenApiParameter(
                "token",
                OpenApiTypes.STR,
                OpenApiParameter.PATH,
                description="Authentication token.",
            )
        ],
        request=SystemSettingsSerializer,
        responses={
            200: SystemSettingsSerializer,
            400: ErrorResponseSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        },
        tags=["System Settings"],
    )
    def patch(self, request, token, format=None):
        logger.info("Updating system settings for user: %s", request.validated_user.username)
        try:
            settings_obj, _ = SystemSettings.objects.get_or_create(id=1)
            serializer = SystemSettingsSerializer(
                settings_obj, data=request.data, partial=True
            )
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error updating system settings: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AuditLogView(APIView):
    """
    Retrieves a list of audit log entries.
    Supports pagination. Access restricted by role.
    """

    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="List Audit Logs",
        description=(
            "Fetches a paginated list of audit log entries. Access restricted to "
            "'admin' and 'reviewer' roles."
        ),
        parameters=[
            OpenApiParameter(
                "token",
                OpenApiTypes.STR,
                OpenApiParameter.PATH,
                description="Authentication token.",
            ),
            OpenApiParameter(
                "page",
                OpenApiTypes.INT,
                OpenApiParameter.QUERY,
                description="Page number for pagination.",
            ),
        ],
        responses={
            200: AuditLogSerializer(many=True),
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        },
        tags=["Audit Log"],
    )
    def get(self, request, token, format=None):
        logger.info("Fetching audit logs for user: %s", request.validated_user.username)
        try:
            if request.validated_user.role not in ["admin", "reviewer"]:
                return Response(
                    {"error": "Forbidden. You do not have permission to view audit logs."},
                    status=status.HTTP_403_FORBIDDEN,
                )

            logs = AuditLog.objects.all().order_by("-timestamp")

            paginator = PageNumberPagination()
            paginated_queryset = paginator.paginate_queryset(logs, request, view=self)
            serializer = AuditLogSerializer(paginated_queryset, many=True)
            return paginator.get_paginated_response(serializer.data)
        except Exception as e:
            logger.error(f"Error fetching audit logs with pagination: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UserProfileDetailsView(APIView):
    """
    API to fetch and update user profile details based on the provided token.
    """

    permission_classes = [RoleBasedPermission]

    @extend_schema(
        summary="Get User Profile",
        description=(
            "Fetches the details of the authenticated user's profile based on the "
            "provided token."
        ),
        parameters=[
            OpenApiParameter(
                "token",
                OpenApiTypes.STR,
                OpenApiParameter.PATH,
                description="Authentication token.",
            )
        ],
        responses={
            200: UserProfileSerializer,
            401: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
        },
        tags=["User Profile"],
    )
    def get(self, request, token):
        user_data = token_verification(token)
        if user_data["status"] != 200:
            return Response(
                {"error": user_data["error"]}, status=status.HTTP_400_BAD_REQUEST
            )

        user = user_data["user"]
        try:
            user_profile = UserProfile.objects.get(username=user)
            serializer = UserProfileSerializer(user_profile)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except UserProfile.DoesNotExist:
            return Response(
                {"error": "User profile not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            logger.error(f"Error fetching user profile: {str(e)}")
            return Response(
                {"error": "An error occurred while fetching the user profile."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @extend_schema(
        summary="Update User Profile",
        description="Update the user's profile information such as name and theme.",
        request=UserProfileSerializer,
        responses={
            200: UserProfileSerializer,
            400: ErrorResponseSerializer,
            404: ErrorResponseSerializer,
            403: ErrorResponseSerializer,
        },
        tags=["User Profile"],
    )
    def put(self, request, token):
        user_data = token_verification(token)
        if user_data["status"] != 200:
            return Response(
                {"error": user_data["error"]}, status=status.HTTP_400_BAD_REQUEST
            )

        user = user_data["user"]
        try:
            user_profile = UserProfile.objects.get(username=user)
        except UserProfile.DoesNotExist:
            return Response(
                {"error": "User profile not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            logger.error(f"Error fetching user profile for update: {str(e)}")
            return Response(
                {"error": "An error occurred while fetching the user profile."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        serializer = UserProfileSerializer(user_profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

