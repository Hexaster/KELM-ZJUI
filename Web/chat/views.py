from django.shortcuts import render
from django.http import JsonResponse
import sys
import os
from pathlib import Path
from LLM import LLM_loader

# Add the LLM module to the Python path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LLM_DIR = os.path.join(BASE_DIR, "LLM")
sys.path.append(str(LLM_DIR))

llm = LLM_loader()

# Create your views here.
def chat_view(request):
    messages = request.session.get('chat_messages', [])
    if request.method == 'POST':
        user_message = request.POST.get('message')
        if user_message:
            messages.append({'text': user_message, 'user': True})
            request.session['chat_messages'] = messages

    return render(request, 'chat/chat.html', {'messages': messages})
