from django.shortcuts import render, redirect
from LLM import LLM_loader

llm = None

# Create your views here.
def chat_view(request):
    chat_history = request.session.get('chat_messages', []) #Get chat history
    print(chat_history)
    if request.method == 'POST' and 'clear-button' in request.POST:
        # If "clear" was hit, clear everything
        request.session.flush()
        return render(request, 'chat/chat.html', {'messages': chat_history})

    if request.method == 'POST' and 'send-button' in request.POST:
        user_message = request.POST.get('message')
        if user_message:
            chat_history.append({'text': user_message, 'role': "user"})
            response = get_response(chat_history, user_message)
            #print(response)
            chat_history.append({'text': response, 'role': "assistant"})
            request.session['chat_messages'] = chat_history
    return render(request, 'chat/chat.html', {'messages': chat_history})

def get_llm():
    global llm
    if llm is None:
        llm = LLM_loader.LLM_loader()
    return llm
def get_response(chat_history, messages):
    llm = get_llm()
    llm.set_chat_history(chat_history)
    #print(llm.messages)
    response = llm.generate_response(messages)
    return response