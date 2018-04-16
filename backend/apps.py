from django.apps import AppConfig

from . import views


class BackendConfig(AppConfig):
    name = 'backend'
    verbose_name = "Neural Politician Backend"

    def ready(self):
        views.init_models()
