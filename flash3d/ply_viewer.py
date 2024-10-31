import gradio as gr
import os
root_directory = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(root_directory, 'rotate_demo/demo.ply')

def display_ply(ply_file):

    # return "/media/wjq/wjq/3dgs_learn/3dgs_output/output/point_cloud/iteration_7000/point_cloud.ply"
    return output_path

css = """
h1 {
    text-align: center;
    display: block;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
        # PLY File Viewer
        **Upload a .ply file to view the 3D model.**
        """
    )

    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            input_ply = gr.File(label="Upload PLY File", file_types=[".ply"], elem_id="ply_file")
            submit = gr.Button("Display", elem_id="display", variant="primary")
        with gr.Column(scale=2):
            output_model = gr.Model3D(label="Output Model", height=640, interactive=False)
    
    submit.click(
        fn=display_ply,
        inputs=[input_ply],
        outputs=[output_model]
    )

demo.launch()
