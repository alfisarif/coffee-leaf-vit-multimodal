"""Inference entry point for a saved checkpoint. See docs/reproduction_protocol.md."""
import argparse

def main():
    p=argparse.ArgumentParser(); p.add_argument('--experiment',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--images-dir',required=True); p.add_argument('--output',required=True); args=p.parse_args()
    raise NotImplementedError('Use the training/evaluation pipeline as documented; this entry point is reserved for checkpoint inference.')
if __name__=='__main__': main()
