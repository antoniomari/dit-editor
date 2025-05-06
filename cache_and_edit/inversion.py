from cache_and_edit import CachedPipeline

def image2latent(pipe, image, latent_nudging_scalar = 1.15):
    image = pipe.image_processor.preprocess(image).type(pipe.vae.dtype).to("cuda")
    latents = pipe.vae.encode(image)["latent_dist"].mean
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    latents = latents * latent_nudging_scalar
    latents = pipe._pack_latents(
        latents=latents,
        batch_size=1,
        num_channels_latents=16,
        height=128,
        width=128
    )

    return latents


def get_inverted_input_noise(pipe: CachedPipeline, image, num_steps: int = 28):
    """_summary_

    Args:
        pipe (CachedPipeline): _description_
        image (_type_): _description_
        num_steps (int, optional): _description_. Defaults to 28.

    Returns:
        _type_: _description_
    """

    noise = pipe.run(
        "",
        num_inference_steps=num_steps,
        seed=42,
        guidance_scale=1.5,
        output_type="latent",
        latents=image2latent(pipe.pipe, image),
        inverse=True,
    ).images[0]

    return noise
