
from django.shortcuts import render
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
#sys.path.append(os.path.join(BASE_DIR, 'KELM/LLM'))
from LLM import LLM_loader

llm = None

# Create your views here.
def chat_view(request):
    messages = request.session.get('chat_messages', []) #Get chat history
    if request.method == 'POST':
        user_message = request.POST.get('message')
        if user_message:
            messages.append({'text': user_message, 'user': True})
            #response = get_response(user_message)
            #print(response)
            #messages.append({'text': response, 'user': False})
            request.session['chat_messages'] = messages
    return render(request, 'chat/chat.html', {'messages': messages})

def get_llm():
    global llm
    if llm is None:
        llm = LLM_loader.LLM_loader()
    return llm
def get_response(messages):
    llm = get_llm()
    response = llm.generate_response(messages)
    return response

if __name__ == "__main__":
    print()