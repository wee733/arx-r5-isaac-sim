"""Use the pinned official CLI with checkpointing and full-parameter evidence."""
import json
import os
from pathlib import Path
import runpy

from gr00t.experiment import experiment
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7
from huggingface_hub import snapshot_download
from transformers import TrainerCallback

rank = int(os.environ.get('RANK', '0'))
evidence = Path(os.environ['B2ARX_EVIDENCE'])
original_create = Gr00tN1d7Pipeline._create_model
original_run = experiment.run
original_processor = processing_gr00t_n1d7.build_processor

def local_processor(model_name, transformers_loading_kwargs):
    # Transformers 4.57's tokenizer regex check ignores local_files_only for
    # Hub IDs. An absolute snapshot path avoids that extra metadata request.
    local_path = snapshot_download(model_name, local_files_only=True) if not Path(model_name).is_dir() else model_name
    return original_processor(local_path, transformers_loading_kwargs)

processing_gr00t_n1d7.build_processor = local_processor

class EarlyCheckpoint(TrainerCallback):
    """Save once at step 10, then retain the configured 200-step cadence."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 10:
            control.should_save = True
        return control

original_trainer_init = experiment.Gr00tTrainer.__init__

def audited_trainer_init(self, *args, **kwargs):
    original_trainer_init(self, *args, **kwargs)
    self.add_callback(EarlyCheckpoint())

experiment.Gr00tTrainer.__init__ = audited_trainer_init

def write(name, data):
    if rank == 0:
        evidence.mkdir(parents=True, exist_ok=True)
        target = evidence / name
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(target)

def audited_create(self):
    model = original_create(self)
    modules = {}
    frozen = []
    adapters = []
    groups = {
        'language': '.language_model.',
        'vision': '.visual.',
        'projector': 'action_head.action_decoder.',
        'diffusion': 'action_head.model.',
    }
    gradients = {}
    for name, parameter in model.named_parameters():
        group = next((key for key, prefix in groups.items() if prefix in name), 'other')
        entry = modules.setdefault(group, {'total': 0, 'trainable': 0})
        entry['total'] += parameter.numel()
        entry['trainable'] += parameter.numel() if parameter.requires_grad else 0
        if not parameter.requires_grad:
            frozen.append(name)
        if 'lora_' in name.lower():
            adapters.append(name)
    report = {'modules': modules, 'frozen_parameters': frozen, 'lora_parameters': adapters,
              'total': sum(x['total'] for x in modules.values()),
              'trainable': sum(x['trainable'] for x in modules.values())}
    write('trainable_parameters.json', report)
    assert not frozen and not adapters, 'Expected all original model parameters trainable without LoRA'
    assert all(modules.get(key, {}).get('trainable', 0) > 0 for key in groups)
    # Observe one real backward gradient in each major module, without changing it.
    if rank == 0:
        for key, prefix in groups.items():
            name, parameter = next((n, p) for n, p in model.named_parameters()
                                   if prefix in n and p.ndim >= 2)
            def capture(grad, key=key, name=name):
                if key not in gradients:
                    norm = float(grad.detach().float().norm().item())
                    gradients[key] = {'parameter': name, 'gradient_l2': norm}
                    write('first_backward_gradients.json', gradients)
                return grad
            parameter.register_hook(capture)
    return model

def configured_run(config):
    assert all(getattr(config.model, key) for key in
               ('tune_llm', 'tune_visual', 'tune_projector', 'tune_diffusion_model'))
    config.model.tune_vlln = True
    config.training.gradient_checkpointing = True
    config.training.transformers_local_files_only = True
    write('launch_settings.json', {
        'official_commit': 'dac54b313160c201c450f97df0fd74ebd379c892',
        'gradient_checkpointing': True, 'tune_vlln': True,
        'early_checkpoint_step': 10,
        'max_steps': config.training.max_steps,
        'global_batch_before_accumulation': config.training.global_batch_size,
        'gradient_accumulation_steps': config.training.gradient_accumulation_steps,
        'effective_batch': config.training.global_batch_size * config.training.gradient_accumulation_steps,
        'num_gpus': config.training.num_gpus, 'learning_rate': config.training.learning_rate,
        'datasets': [str(x) for x in config.data.datasets],
    })
    return original_run(config)

Gr00tN1d7Pipeline._create_model = audited_create
experiment.run = configured_run
runpy.run_path(str(Path(os.environ['GR00T_REPO']) / 'gr00t/experiment/launch_finetune.py'), run_name='__main__')
