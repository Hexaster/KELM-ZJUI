from django.shortcuts import render

# Create your views here.
def chat_view(request):
    messages = request.session.get('chat_messages', [])
    if request.method == 'POST':
        user_message = request.POST.get('message')
        if user_message:
            messages.append({'text': user_message, 'user': True})
            request.session['chat_messages'] = messages

    return render(request, 'chat/chat.html', {'messages': messages})
