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
                'sopcreateview': ['post'],
                'soplistview': ['get'],
                'sopdetailview': ['get', 'put', 'delete'], 
                'audiofiledetailview': ['get', 'delete'],
                'processaudioview': ['post'],
                'feedbackview': ['post'], 
                'feedbacklistview': ['get'], 
                'feedbackdetailview': ['get', 'put', 'patch', 'delete'], 
                'getaudiorecordsview': ['get'],
                'reanalyzeaudioview': ['post'],
                'sessioncreateview': ['post'],
                'sessionlistview': ['get'],
                'sessiondetailview': ['get', 'put', 'patch', 'delete'], 
                'feedbackreviewlistview': ['get'], # Added FeedbackReviewListView
                'feedbackreviewdetailview': ['get'],      # Added FeedbackReviewDetailView (Operator GET only)
                'usersettingsview': ['get', 'patch'],
                'sessionstatusupdateview': ['patch'], 
                'adminuserlistview': ['get']
            },
            'reviewer': {
                'soplistview': ['get'],
                'sopdetailview': ['get'], 
                'audiofiledetailview': ['get'],
                'feedbacklistview': ['get'], 
                'feedbackdetailview': ['get'], 
                'getaudiorecordsview': ['get'],
                'sessionlistview': ['get'],
                'sessiondetailview': ['get'], 
                'sessionreviewview': ['get', 'post'], # Existing permission for creating reviews
                'feedbackreviewlistview': ['get'], # Added FeedbackReviewListView
                'feedbackreviewdetailview': ['get', 'put', 'patch', 'delete'], # Added FeedbackReviewDetailView
                'auditlogview': ['get'],
            }
        }

        allowed_methods = permissions.get(role, {}).get(view_name, [])
        is_allowed = method in allowed_methods
        logger.info(f"Permission check result: {'Allowed' if is_allowed else 'Denied'}")
        return is_allowed