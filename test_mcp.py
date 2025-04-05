# import requests
#
# # Simple GET request to the SSE endpoint
# response = requests.get("http://localhost:8002/sse", stream=True)
#
# # Read and print the raw data that contains the session ID
# for line in response.iter_lines(decode_unicode=True):
#     if line.startswith("data: ") and "session_id" in line:
#         print(line)
#         # The session ID is embedded in this line
#         break
#
# exit()

import requests

response = requests.post(
    "http://localhost:8002/messages/messages/?session_id=75470c8d4fa04289937e9f4ad3dc2f45",
    json={
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "request_courses",
            "arguments": {
                "query_string": "Machine Learning for Executives",
                "k": 5,
                "n_results": 30,
            },
        },
    },
)
print(response.status_code)
print(response.json())
