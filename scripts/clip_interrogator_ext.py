import gradio as gr
import open_clip
import clip_interrogator
from clip_interrogator import Config, Interrogator

from modules import devices, script_callbacks

__version__ = '0.0.1'

ci = None

def load(clip_model_name):
    global ci
    if ci is None:
        print(f"Loading CLIP Interrogator {clip_interrogator.__version__}...")
        ci = Interrogator(Config(device=devices.get_optimal_device(), clip_model_name=clip_model_name))
    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        ci.load_clip_model()
    ci.blip_model = ci.blip_model.to(devices.get_optimal_device())
    ci.clip_model = ci.clip_model.to(devices.get_optimal_device())

def unload():
    global ci
    if ci is not None:
        print("Offloading CLIP Interrogator...")
        ci.blip_model = ci.blip_model.to("cpu")
        ci.clip_model = ci.clip_model.to("cpu")
        devices.torch_gc()

def get_models():
    return ['/'.join(x) for x in open_clip.list_pretrained()]

def image_analysis(image, clip_model_name):
    load(clip_model_name)

    image = image.convert('RGB')
    image_features = ci.image_to_features(image)

    top_mediums = ci.mediums.rank(image_features, 5)
    top_artists = ci.artists.rank(image_features, 5)
    top_movements = ci.movements.rank(image_features, 5)
    top_trendings = ci.trendings.rank(image_features, 5)
    top_flavors = ci.flavors.rank(image_features, 5)

    medium_ranks = {medium: sim for medium, sim in zip(top_mediums, ci.similarities(image_features, top_mediums))}
    artist_ranks = {artist: sim for artist, sim in zip(top_artists, ci.similarities(image_features, top_artists))}
    movement_ranks = {movement: sim for movement, sim in zip(top_movements, ci.similarities(image_features, top_movements))}
    trending_ranks = {trending: sim for trending, sim in zip(top_trendings, ci.similarities(image_features, top_trendings))}
    flavor_ranks = {flavor: sim for flavor, sim in zip(top_flavors, ci.similarities(image_features, top_flavors))}
    
    return medium_ranks, artist_ranks, movement_ranks, trending_ranks, flavor_ranks

def image_to_prompt(image, mode, clip_model_name):
    load(clip_model_name)

    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    elif mode == 'fast':
        return ci.interrogate_fast(image)
    elif mode == 'negative':
        return ci.interrogate_negative(image)

def prompt_tab():
    with gr.Column():
        with gr.Row():
            image = gr.Image(type='pil', label="Image")
            with gr.Column():
                mode = gr.Radio(['best', 'fast', 'classic', 'negative'], label='Mode', value='best')
                model = gr.Dropdown(get_models(), value='ViT-L-14/openai', label='CLIP Model')
        prompt = gr.Textbox(label="Prompt")
    with gr.Row():
        button = gr.Button("Generate", variant='primary')
        unload_button = gr.Button("Unload")
    button.click(image_to_prompt, inputs=[image, mode, model], outputs=prompt)
    unload_button.click(unload)

def analyze_tab():
    with gr.Column():
        with gr.Row():
            image = gr.Image(type='pil', label="Image")
            model = gr.Dropdown(get_models(), value='ViT-L-14/openai', label='CLIP Model')
        with gr.Row():
            medium = gr.Label(label="Medium", num_top_classes=5)
            artist = gr.Label(label="Artist", num_top_classes=5)        
            movement = gr.Label(label="Movement", num_top_classes=5)
            trending = gr.Label(label="Trending", num_top_classes=5)
            flavor = gr.Label(label="Flavor", num_top_classes=5)
    button = gr.Button("Analyze", variant='primary')
    button.click(image_analysis, inputs=[image, model], outputs=[medium, artist, movement, trending, flavor])

def about_tab():
    gr.Markdown("## 🕵️‍♂️ CLIP Interrogator 🕵️‍♂️")
    gr.Markdown("*Want to figure out what a good prompt might be to create new images like an existing one? The CLIP Interrogator is here to get you answers!*")
    gr.Markdown("## Notes")
    gr.Markdown(
        "* For best prompts with Stable Diffusion 1.* choose the **ViT-L-14/openai** model.\n"
        "* For best prompts with Stable Diffusion 2.* choose the **ViT-H-14/laion2b_s32b_b79k** model.\n"
        "* When you are done click the **Unload** button to free up memory."
    )
    gr.Markdown("## Github")
    gr.Markdown("If you have any issues please visit [CLIP Interrogator on Github](https://github.com/pharmapsychotic/clip-interrogator) and drop a star if you like it!")
    gr.Markdown(f"<br><br>CLIP Interrogator version: {clip_interrogator.__version__}<br>Extension version: {__version__}")

def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tab("Prompt"):
            prompt_tab()
        with gr.Tab("Analyze"):
            analyze_tab()
        with gr.Tab("About"):
            about_tab()

    return [(ui, "Interrogator", "interrogator")]

script_callbacks.on_ui_tabs(add_tab)