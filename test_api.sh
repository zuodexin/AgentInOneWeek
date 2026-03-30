curl http://10.0.0.114/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ${GPUSTACK_API_KEY}" \
-d '{
  "seed": null,
  "stop": null,
  "temperature": 1,
  "top_p": 1,
  "max_tokens": 16384,
  "frequency_penalty": 0,
  "presence_penalty": 0,
  "model": "qwen3.5-0.8b",
  "messages": [
    {
      "role": "user",
      "content": "你好"
    }
  ]
}'