from django.contrib import admin
from django.urls import path, include, re_path
from django.contrib.staticfiles.views import serve
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('diagnostic.urls')),

    re_path(
        r'^media/(?P<path>.*)$',
        serve,
        {
            'insecure': True,
            'document_root': settings.MEDIA_ROOT,
        }
    ),
]