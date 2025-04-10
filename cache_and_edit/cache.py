class FluxActivationCache:

    NUM_TRANSFORMER_BLOCKS = 19
    NUM_SINGLE_TRANSFORMER_BLOCKS = 38
    
    def __init__(self):
        # For "double" transformer
        self.image_residual = [None] * FluxActivationCache.NUM_TRANSFORMER_BLOCKS
        self.image_activation = [None] * FluxActivationCache.NUM_TRANSFORMER_BLOCKS
        self.text_residual = [None] * FluxActivationCache.NUM_TRANSFORMER_BLOCKS
        self.text_activation = [None] * FluxActivationCache.NUM_TRANSFORMER_BLOCKS

        # For single transformer
        self.text_image_residual = [None] * FluxActivationCache.NUM_SINGLE_TRANSFORMER_BLOCKS
        self.text_image_activation = [None] * FluxActivationCache.NUM_SINGLE_TRANSFORMER_BLOCKS


# TODO: build it similarly to FluxActivationCache
class PixartActivationCache:

    NUM_TRANSFORMER_BLOCKS = 28
    
    def __init__(self):
        # For "double" transformer
        ... 

