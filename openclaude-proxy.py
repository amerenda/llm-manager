def proxy(endpoint, request_data):
    """Forward request to backend."""
    try:
        resp = requests.post(
            endpoint + "/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=180
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": f"HTTP {resp.status_code}", "detail": resp.text}
    except requests.Timeout:
        return {"error": "timeout", "detail": "Server timed out after 180s"}
    except requests.ConnectionError as e:
        return {"error": "connection_error", "detail": str(e)}


def route_request(request_data):
    """Route to appropriate backend."""
    user_prompt = request_data["messages"][0]["content"]
    model = analyze_prompt(request_data["messages"], user_prompt)

    endpoint = MURDERBOT_URL if model == "murderbot" else OLLAMA_URL
    return proxy(endpoint, request_data)


@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible routing endpoint."""
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "No request body"}), 400

        result = route_request(request_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "internal_error", "detail": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("DEBUG", "true").lower() == "true")
