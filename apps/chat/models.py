from django.db import models

# Create your models here.
class ChatMessage(models.Model):
    session_id = models.CharField(max_length=50) #identifier of session/user
    role = models.CharField(max_length=10) #user or agent
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.role} ({self.timestamp}): {self.content[:50]}"
