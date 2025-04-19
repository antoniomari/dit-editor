import gc
from diffusers import FluxPipeline
import torch
from cache_and_edit.cache import FluxActivationCache, ActivationCacheEditor

class CachedPipeline:
    
    def __init__(self, pipe: FluxPipeline):
        self.pipe = pipe
        self.cache = None
        self.cache_editor = None
    
    @torch.no_grad
    def run(self, 
            prompt: str, 
            num_inference_steps: int = 1,
            seed: int = 42):
        
        if self.cache:
            self.cache_editor.clear_cache_hooks()
            del(self.cache_editor)
            del(self.cache)
            gc.collect()              # force Python to clean up unreachable objects
            torch.cuda.empty_cache()  # tell PyTorch to release unused GPU memory from its cache
            
        self.cache = FluxActivationCache()
        self.cache_editor = ActivationCacheEditor(self.cache)
        
        self.cache_editor.set_cache_hooks(self.pipe)
        return self.pipe(prompt=prompt, 
                         num_inference_steps=num_inference_steps,
                         generator=torch.Generator(device="cpu").manual_seed(seed))
