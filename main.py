import yaml
import wandb
from datetime import datetime
import argparse
import torch
from networks import RecurrentBackbone, LinearAttentionBackbone, HopfieldAttentionBackbone
from benchmarks import LAMBADA, WikiText, LRA, MemoryCopying

def run_experiment(conf):
    match conf['model']['name']:
        case 'rnn':
            backbone = RecurrentBackbone(conf['model']['dim'], 
                                         conf['model']['num_layers']).to(conf['hardware']['device'])
        case 'linear_attention':
            backbone = LinearAttentionBackbone(conf['model']['dim'], 
                                               conf['model']['num_layers']).to(conf['hardware']['device'])
        case 'hopfield_attention':
            backbone = HopfieldAttentionBackbone(conf['model']['dim'], 
                                                 conf['model']['num_layers']).to(conf['hardware']['device'])

    match conf['benchmark']['name']:
        case 'lambada':
            benchmark = LAMBADA(conf)
        case 'wikitext':
            benchmark = WikiText(conf)
        case 'lra':
            benchmark = LRA(conf)
        case 'memcopy':
            benchmark = MemoryCopying(conf)

    benchmark.run_benchmark(backbone)    

def recursive_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            recursive_update(base[key], value)
        else:
            base[key] = value
    return base

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--sweep')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        print("Running in DEBUG mode. Wandb logging will not occur.")
        from datasets import disable_caching; disable_caching() # disable dataset caching on map

    with open(args.config, 'rb') as f:
        conf = yaml.safe_load(f.read())
        conf['debug'] = args.debug

    if args.sweep and not args.debug:
        with open(args.sweep, 'rb') as f:
            sweep_conf = yaml.safe_load(f.read())

        sweep_id = wandb.sweep(sweep_conf, project=sweep_conf["project"])

        # Define sweep run function
        def sweep_experiment():
            _ = wandb.init()
            exp_conf = dict(wandb.config)
            merged_conf = recursive_update(conf.copy(), exp_conf)
            run_experiment(merged_conf)

        # Launch sweep agent
        wandb.agent(sweep_id, function=sweep_experiment)
    else:
        if conf['hardware']['device'] == 'cuda':
            assert torch.cuda.is_available()

        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run = wandb.init(
            project="language_model_benchmarks",
            name=f"{conf['benchmark']['name']}_{conf['model']['name']}_{datetime_str}",
            config=conf,
            group=conf['benchmark']['name'],
            tags=[conf['benchmark']['name'], conf['model']['name']],
            mode='disabled' if args.debug else 'online'
        )
        run_experiment(conf)
        wandb.finish()
        
    
