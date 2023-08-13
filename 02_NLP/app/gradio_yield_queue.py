import gradio as gr
import time
import numpy as np


def fake_diffusion(steps):
    s = "Intrusion Detection Evaluation Dataset"
    output = ""
    for i in s.split(" "):
        time.sleep(1)
        output += " " + i
        yield output
        # return output


# 設置滑窗，最小值為1，最大值為10，初始值為3，每次改動增減1位
demo = gr.Interface(fake_diffusion, inputs=gr.Slider(1, 10, value=3, step=1), outputs="text")
# 生成器必須要queue函數
demo.queue().launch(server_name="0.0.0.0", share=True)
