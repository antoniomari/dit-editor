class HooksRegister:
    
    @staticmethod
    def _register_general_hook(pipe, position, hook, with_kwargs=False, is_pre_hook=False):

        block = HooksRegister._locate_block(pipe, position)

        if is_pre_hook:
            return block.register_forward_pre_hook(hook, with_kwargs=with_kwargs)
        else:
            return block.register_forward_hook(hook, with_kwargs=with_kwargs)

    @staticmethod
    def _locate_block(pipe, position: str):
        '''
        Locate the block at the specified position in the pipeline.
        '''
        block = pipe
        for step in position.split('.'):
            if step.isdigit():
                step = int(step)
                block = block[step]
            else:
                block = getattr(block, step)
        return block