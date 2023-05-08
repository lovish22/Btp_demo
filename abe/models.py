from django.db import models
from asyncio.windows_events import NULL


class audio(models.Model):
    audio_name = models.CharField(max_length=255, primary_key=True)
    audio_file = models.FileField(upload_to="audio_files")

    def __str__(self):
        return self.audio_name
