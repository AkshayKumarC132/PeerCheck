from rest_framework import permissions
from .authentication import token_verification
import logging

logger = logging.getLogger(__name__)

class RoleBasedPermission(permissions.BasePermission):
    def has_permission(self, request, view):
        logger.info(f"Checking permissions for view: {view.__class__.__name__}")
        
        # Extract token from URL kwargs
        token = view.kwargs.get('token')
        if not token:
            logger.error("No token provided in URL")
            return False

        # Validate token
        user_data = token_verification(token)
        if user_data['status'] != 200:
            logger.error(f"Token validation failed: {user_data['error']}")
            return False

        # Store validated user in request for view logic
        request.validated_user = user_data['user']
        role = user_data['user'].role
        method = request.method.lower()
        view_name = view.__class__.__name__.lower()

        logger.info(f"User: {user_data['user'].username}, Role: {role}, Method: {method}, View: {view_name}")

        # Admin has full access
        if role == 'admin':
            logger.info("Admin role granted full access")
            return True

        # Define permissions per role
        permissions = {
            'operator': {
                'processaudioview': ['post'],
                'sessioncreateview': ['post'],
                'sessionlistview': ['get'],
                'soplistview': ['get'],
                'usersettingsview': ['get', 'patch'],
                'feedbackview': ['post'], # Retained as per instruction
                'getaudiorecordsview': ['get'],
                'reanalyzeaudioview': ['post'],
                # SOPCreateView is admin only now
            },
            'reviewer': {
                'sessionlistview': ['get'],
                'sessionreviewview': ['get', 'post'],
                'soplistview': ['get'],
                'getaudiorecordsview': ['get'], # Access to relevant records
                'usersettingsview': ['get', 'patch'],
                'auditlogview': ['get'],
            }
            # Admin specific views like sopcreateview, systemsettingsview, full auditlogview
            # are covered by the "if role == 'admin': return True" check.
            # If more granular control for admin is needed below this block,
            # it can be added, but typically admin has all permissions not explicitly denied.
        }

        # Specific views for admin not covered by operator/reviewer roles explicitly listed
        # This is mostly for documentation or if the admin check was more nuanced.
        # Given the current "return True" for admin, this section is not strictly necessary for functionality
        # but helps in defining what an admin *can* do if that logic were different.
        if role == 'admin':
            admin_permissions = {
                'sopcreateview': ['post', 'get'], # get is usually via soplistview
                'systemsettingsview': ['get', 'patch'],
                'auditlogview': ['get'], # Already in reviewer, but admin also has it
                # Plus all operator and reviewer permissions
            }
            # Merge admin specific perms if needed, or rely on the blanket True
            # For now, the blanket True for admin covers these.

        allowed_methods = permissions.get(role, {}).get(view_name, [])
        is_allowed = method in allowed_methods
        logger.info(f"Permission check result: {'Allowed' if is_allowed else 'Denied'}")
        return is_allowed