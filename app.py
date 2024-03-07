import gradio as gr
from Blur import background_motion_blur


def blur(img, distance_blur, amount_blur):
    return background_motion_blur(img, distance_blur, amount_blur)


iface = gr.Interface(fn=blur,
                     title="Background Motion Blur",
                     examples=[["input.jpg", 200, 1]],

                     inputs=[gr.Image(type='pil', label='Image'),
                             gr.Slider(label='Blur Distance', minimum=0, maximum=500, value=100),
                             gr.Slider(label='Blur Amount', minimum=0.0, maximum=1.0, value=0.75)],

                     outputs=gr.Image(label='Output'))


iface.launch()
