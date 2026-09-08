import argparse, subprocess, sys

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--images-dir',required=True); args=ap.parse_args()
    exps=['sensor_only','vit_only','vit_sensor','shuffled_sensor','efficientnet_b0','maxvit_tiny']
    subprocess.run([sys.executable,'-m','src.verify_dataset','--images-dir',args.images_dir],check=True)
    for exp in exps: subprocess.run([sys.executable,'-m','src.train','--experiment',exp,'--images-dir',args.images_dir],check=True)
    subprocess.run([sys.executable,'-m','src.sensor_diagnostic'],check=True)
if __name__=='__main__': main()
