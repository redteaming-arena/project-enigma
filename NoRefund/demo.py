import gradio as gr
from test_customer_agent import DemoSession

# Initialize chat history
initial_history = [
    {"role": "assistant", "content": "Hello! What can I help you today."},
]

if __name__ == "__main__":
    model = 'accounts/fireworks/models/llama-v3p1-70b-instruct'

    session = DemoSession(model=model)

    # Create the Gradio interface
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(
            value=initial_history,
            type="messages",
            height=400
        )
        msg = gr.Textbox(
            label="Type your message here",
            placeholder="Type your message and press enter",
            show_label=False
        )
        init = gr.Button("Initialize")
        # clear = gr.Button("Clear")

        def user(user_message, history):
            """Handle user messages and update chat history."""
            return "", history + [{'role': 'user', 'content': user_message}]

        def bot(history):
            """Generate bot response and update chat history."""
            process_history = [{
                'role': a['role'],
                'content': a['content_real'] if 'content_real' in a else a['content']
            } for a in history]
            bot_message, functions = session.generate(process_history)

            display_message = bot_message
            for k, v in functions.items():
                display_message += f'\n[FUNCTION CALLED]: {v["name"]}({v["arguments"]})'

            history.append({
                'role': 'assistant',
                # "content" also displays function calls
                'content': display_message,
                # "content_real" only includes responses without function calls
                'content_real': bot_message,
            })

            return history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot)

        def initialize():
            session.initialize()
            return initial_history

        init.click(initialize, None, chatbot, queue=False)
        # clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch()