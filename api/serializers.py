"""Serializers used by the active API endpoints."""

from rest_framework import serializers

from .models import AudioFile, AuditLog, SystemSettings, UserProfile, UserSettings


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = [
            "id",
            "email",
            "name",
            "theme",
            "password",
            "role",
            "date_joined",
            "last_login",
        ]
        extra_kwargs = {
            "password": {"write_only": True},
            "date_joined": {"read_only": True},
            "last_login": {"read_only": True},
        }

    def create(self, validated_data):
        password = validated_data.pop("password", None)
        user = UserProfile(**validated_data)
        if password:
            user.set_password(password)
        user.save()
        return user


class AdminUserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = [
            "id",
            "username",
            "email",
            "name",
            "role",
            "is_active",
            "is_staff",
            "theme",
            "date_joined",
            "last_login",
        ]
        read_only_fields = ["date_joined", "last_login", "theme"]

    def update(self, instance, validated_data):
        validated_data.pop("password", None)

        if "role" in validated_data and validated_data["role"] not in [
            choice[0] for choice in UserProfile.ROLE_CHOICES
        ]:
            raise serializers.ValidationError({"role": "Invalid role selected."})

        return super().update(instance, validated_data)


class UserSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserSettings
        fields = ["language", "notification_prefs", "theme", "created_at", "updated_at"]


class SystemSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = SystemSettings
        fields = ["default_sop_version", "timeout_threshold", "created_at", "updated_at"]


class AuditLogSerializer(serializers.ModelSerializer):
    user = UserProfileSerializer(read_only=True)
    session = serializers.PrimaryKeyRelatedField(read_only=True)

    class Meta:
        model = AuditLog
        fields = [
            "id",
            "action",
            "user",
            "timestamp",
            "session",
            "object_id",
            "object_type",
            "details",
        ]


class LoginSerializer(serializers.Serializer):
    email = serializers.CharField()
    password = serializers.CharField()


class AudioFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioFile
        fields = "__all__"
        depth = 1
