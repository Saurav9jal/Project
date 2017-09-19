from django.conf.urls import include, url
from django.contrib import admin
from khoj import views
urlpatterns = [
    # Examples:
    url(r'^$', views.khoj, name='khoj'),
    # url(r'^blog/', include('blog.urls')),

    # url(r'^admin/', include(admin.site.urls)),
]
