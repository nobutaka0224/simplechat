# lambda/index.py
import json
import os
import re
import urllib.request
import urllib.error

def extract_region_from_arn(arn):
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    return "us-east-1"

MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")

BEDROCK_RUNTIME_INVOKE_PATH = f"/model/{MODEL_ID}/invoke"

def lambda_handler(event, context):
    region = extract_region_from_arn(context.invoked_function_arn)
    bedrock_endpoint_base = f"https://bedrock-runtime.{region}.amazonaws.com"
    full_bedrock_invoke_url = f"{bedrock_endpoint_base}{BEDROCK_RUNTIME_INVOKE_PATH}"

    print(f"Bedrock invoke URL: {full_bedrock_invoke_url}")
    print(f"Using model: {MODEL_ID}")

    try:
        print("Received event:", json.dumps(event))

        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")

        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])

        print("Processing message:", message)

        messages = conversation_history.copy()

        messages.append({
            "role": "user",
            "content": message
        })

        bedrock_messages = []
        for msg in messages:
            if msg["role"] == "user":
                bedrock_messages.append({
                    "role": "user",
                    "content": [{"text": msg["content"]}]
                })
            elif msg["role"] == "assistant":
                bedrock_messages.append({
                    "role": "assistant",
                    "content": [{"text": msg["content"]}]
                })

        request_payload = {
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": 512,
                "stopSequences": [],
                "temperature": 0.7,
                "topP": 0.9
            }
        }

        print("Calling Bedrock invoke API via urllib.request with payload:", json.dumps(request_payload))

        request_body_bytes = json.dumps(request_payload).encode('utf-8')

        headers = {
            'Content-Type': 'application/json',
        }

        req = urllib.request.Request(full_bedrock_invoke_url, data=request_body_bytes, headers=headers, method='POST')

        try:
            with urllib.request.urlopen(req) as response:
                response_body_bytes = response.read()
                response_body = response_body_bytes.decode('utf-8')

            response_data = json.loads(response_body)
            print("Bedrock response (from urllib.request):", json.dumps(response_data, default=str))

        except urllib.error.HTTPError as e:
            print(f"HTTP Error calling Bedrock: {e.code} - {e.reason}")
            try:
                error_body_bytes = e.read()
                error_body = error_body_bytes.decode('utf-8')
                print(f"Error Body: {error_body}")
                error_data = json.loads(error_body)
                error_message = error_data.get('message', error_data.get('Error', {}).get('Message', str(e)))
            except Exception:
                 error_message = str(e)

            raise Exception(f"Bedrock API HTTP Error {e.code}: {error_message}") from e

        except urllib.error.URLError as e:
            print(f"URL Error calling Bedrock: {e.reason}")
            raise Exception(f"Bedrock API URL Error: {e.reason}") from e

        except Exception as e:
             print(f"An unexpected error occurred during urllib request: {e}")
             raise Exception(f"Unexpected error calling Bedrock API: {str(e)}") from e


        if not response_data.get('output') or not response_data['output'].get('message') or not response_data['output']['message'].get('content'):
             if 'message' in response_data:
                  raise Exception(f"Bedrock response indicates error: {response_data['message']}")
             else:
                raise Exception("No valid response content from the model")


        assistant_response = response_data['output']['message']['content'][0]['text']

        messages.append({
            "role": "assistant",
            "content": assistant_response
        })

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": messages
            })
        }

    except Exception as error:
        print("Caught overall Error:", str(error))

        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }
