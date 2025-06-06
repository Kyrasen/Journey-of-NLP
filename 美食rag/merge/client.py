import gradio as gr
import requests

def main(query):
    query_data = {
        "query":query
    }

    response = requests.post(
        url = "http://127.0.0.1:44444",
        json = query_data,
        headers = {"Content-type":"application/json"}
    )

    if response.status_code == 200:
        reply1 = response.json()["res"]
    else:
        reply1 = None

    return reply1


if __name__ == "__main__":
    iface = gr.Interface(
        fn = main,
        inputs = gr.Textbox(),
        outputs = gr.Markdown()
    )

    iface.launch(
        server_name = "127.0.0.1",
        server_port = 33333
    )

    pass