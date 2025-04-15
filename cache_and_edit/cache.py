from abc import ABC, abstractmethod
from collections import defaultdict
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
from hooks import HooksRegister
import torch

class ModelActivationCache(ABC):
    def __init__(self):
    
        # Initialize caches for "double transformer" blocks using the subclass-defined NUM_TRANSFORMER_BLOCKS
        if hasattr(self, 'NUM_TRANSFORMER_BLOCKS'):
            self.image_residual = [None] * self.NUM_TRANSFORMER_BLOCKS
            self.image_activation = [None] * self.NUM_TRANSFORMER_BLOCKS
            self.text_residual = [None] * self.NUM_TRANSFORMER_BLOCKS
            self.text_activation = [None] * self.NUM_TRANSFORMER_BLOCKS

        # Initialize caches for "single transformer" blocks if defined (using NUM_SINGLE_TRANSFORMER_BLOCKS)
        if hasattr(self, 'NUM_SINGLE_TRANSFORMER_BLOCKS'):
            self.text_image_residual = [None] * self.NUM_SINGLE_TRANSFORMER_BLOCKS
            self.text_image_activation = [None] * self.NUM_SINGLE_TRANSFORMER_BLOCKS

    @abstractmethod
    def get_cache_info(self):
        """
        Return details about the cache configuration.
        Subclasses must implement this to provide info on their transformer block counts.
        """
        pass


class FluxActivationCache(ModelActivationCache):
    # Define number of blocks for double and single transformer caches
    NUM_TRANSFORMER_BLOCKS = 19
    NUM_SINGLE_TRANSFORMER_BLOCKS = 38

    def __init__(self):
        super().__init__()

    def get_cache_info(self):
        return {
            "double_transformer_blocks": self.NUM_TRANSFORMER_BLOCKS,
            "single_transformer_blocks": self.NUM_SINGLE_TRANSFORMER_BLOCKS,
        }


class PixartActivationCache(ModelActivationCache):
    # Define number of blocks for the double transformer cache only
    NUM_TRANSFORMER_BLOCKS = 28

    def __init__(self):
        super().__init__()

    def get_cache_info(self):
        return {
            "double_transformer_blocks": self.NUM_TRANSFORMER_BLOCKS,
        }


class ActivationCacheEditor:

    def __init__(self, cache: ModelActivationCache):
        self.cache = cache
        self.hooks_dict = defaultdict(list)

    @staticmethod
    def _safe_clip(x: torch.Tensor):
        if x.dtype == torch.float16:
            x[torch.isposinf(x)] = 65504
            x[torch.isneginf(x)] = -65504
        return x
    

    @torch.no_grad()
    def cache_residual_and_activation(self, *args):
        """ 
            To be used as a forward hook on a Transformer Block.
            It caches both residual_stream and activation (defined as output - residual_stream).
        """

        if len(args) == 3:
            module, input, output = args
        elif len(args) == 4:
            module, input, kwinput, output = args

        if isinstance(module, FluxTransformerBlock):
            encoder_hidden_states = output[0]            
            hidden_states = output[1]

            self.cache["transformer_blocks"]["image_activation"].append(hidden_states - kwinput["hidden_states"])
            self.cache["transformer_blocks"]["text_activation"].append(encoder_hidden_states - kwinput["encoder_hidden_states"])
            self.cache["transformer_blocks"]["image_residual"].append(kwinput["hidden_states"])
            self.cache["transformer_blocks"]["text_residual"].append(kwinput["encoder_hidden_states"])

        elif isinstance(module, FluxSingleTransformerBlock):
            self.cache["single_transformer_blocks"]["text_image_activation"].append(output - kwinput["hidden_states"])
            self.cache["single_transformer_blocks"]["text_image_residual"].append(kwinput["hidden_states"])
        else:
            raise NotImplementedError(f"Caching not implemented for {type(module)}")
        
    
    def set_cache_hooks(self, pipe):
        
        # insert cache storing in dict
        for block_type, num_layers in self.cache.get_cache_info().items():

            for i in range(num_layers):

                module_name: str = f"transformer.{block_type}.{i}"

                # setup safe torch16 clipping
                safeclip_hook = HooksRegister._register_general_hook(pipe, module_name, 
                                                         ActivationCacheEditor._safe_clip, 
                                                         with_kwargs=True,
                                                         is_pre_hook=False)
                self.hooks_dict[module_name].append(safeclip_hook)

                # register hook for caching
                hook = HooksRegister._register_general_hook(pipe, module_name, 
                                                         self.cache_residual_and_activation, 
                                                         with_kwargs=True,
                                                         is_pre_hook=False)
                
                self.hooks_dict[module_name].append(hook)


    def clear_cache_hooks(self):

        # Remove hooks
        for _, hooks in self.hooks_dict.items():
                for hook in hooks:
                    hook.remove()

        
