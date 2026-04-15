import requests

url = "http://localhost:8003/chat"
payload = {
    "query": "Hello?",
    "k": 1,
    "namespace": "test",
    "user_id": "ab3a2c",
    "agent_id": "d9f466f3-a6e6-49a7-a819-3ff04ec876ae",
    "thread_id": "123",
    "kb_name": "test",
    "system_prompt": "You are a helpful assistant"
}
response = requests.post(url, json=payload)
print(response.status_code)
print(response.text)
