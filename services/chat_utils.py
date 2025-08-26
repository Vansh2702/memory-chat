import os
import json
import boto3
from services.search_utils import get_top_k_context

bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))

def call_claude(query: str):
    context = get_top_k_context(query, k=3)

    messages = [
        {
            "role": "user",
            "content": f"""You are an assistant. Answer the user's question using only the following notes:\n\n{chr(10).join(f'- {c}' for c in context)}\n\nQuestion: {query}"""
        }
    ]

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7
    }

    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        body=json.dumps(payload),
        contentType="application/json"
    )

    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']
