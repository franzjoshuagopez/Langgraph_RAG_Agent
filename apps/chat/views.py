import json
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from apps.chat.rag.agent import run_agent, init_rag
from config.logger import get_logger

logger = get_logger(__name__)

try:

    logger.info("RAG agent initializing...")
    init_rag()
    logger.info("RAG agent initialized")

except Exception as e:

    logger.exception(f"Failed to initialize RAG agent at startup due to: {e}")


# Create your views here.
def rag_agent_home(request):

    return render(request, "chat/rag_agent_home.html")

@csrf_exempt
def send_message(request):
    logger.info("calling send message")

    if request.method == "POST":
        try:

            data = json.loads(request.body) #parse request to JSON format
            user_input = data.get("message", "")
            logger.info("calling RAG agent")
            response = run_agent(user_input) # call RAG agent
            logger.info("RAG agent responded")

            return JsonResponse({"reply" : response})
        
        except Exception as e:
            logger.exception(f"failed due to: {e}")
            return JsonResponse({"error" : str(e)}, status=500)
    
    return JsonResponse({"error" : "Invalid request method"}, status=400)