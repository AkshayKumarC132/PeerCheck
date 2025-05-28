from rest_framework import permissions

class RoleBasedPermission(permissions.BasePermission):
    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False

        role = request.user.role
        method = request.method.lower()
        view_name = view.__class__.__name__

        # Admin has full access
        if role == 'admin':
            return True

        # Define permissions per role
        permissions = {
            'user': {
                'sopcreateview': ['post'],
                'soplistview': ['get'],
                'processaudioview': ['post'],
                'feedbackview': ['post'],
                'getaudiorecordsview': ['get'],
                'reanalyzeaudioview': ['post'],
                'sessioncreateview': ['post'],
                'sessionlistview': ['get'],
            },
            'auditor': {
                'soplistview': ['get'],
                'getaudiorecordsview': ['get'],
                'sessionlistview': ['get'],
            }
        }

        allowed_methods = permissions.get(role, {}).get(view_name.lower(), [])
        return method in allowed_methods