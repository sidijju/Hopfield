import argparse
import torch
from networks import RecurrentBackbone, LinearAttentionBackbone, HopfieldAttentionBackbone
from benchmarks import LAMBADA, WikiText, LRA, MemoryCopying

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--sequence_length', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=256)   
    parser.add_argument('--layers', type=int, default=2)  
    parser.add_argument('--epochs', type=int, default=10)   
    parser.add_argument('--benchmark', choices=['lambada', 'wikitext', 'lra', 'memcopy'], default='lambada')    
    parser.add_argument('--model', choices=['rnn', 'linear_attention', 'hopfield_attention'], default='linear_attention')    
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    match args.model:
        case 'rnn':
            backbone = RecurrentBackbone(args.device, args.hidden, args.layers).to(args.device)
        case 'linear_attention':
            backbone = LinearAttentionBackbone(args.device, args.hidden, args.layers).to(args.device)
        case 'hopfield_attention':
            backbone = HopfieldAttentionBackbone(args.device, args.hidden, args.layers).to(args.device)

    match args.benchmark:
        case 'lambada':
            benchmark = LAMBADA(args)
        case 'wikitext':
            benchmark = WikiText(args)
        case 'lra':
            benchmark = LRA(args)
        case 'memcopy':
            benchmark = MemoryCopying(args)

    benchmark.run_benchmark(backbone)
    
