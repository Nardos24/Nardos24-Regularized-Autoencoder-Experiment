import argparse
import yaml
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["train", "evaluate"], help="Task to perform")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Perform the selected task
    if args.task == "train":
        train_model(config)
    elif args.task == "evaluate":
        evaluate_model(config)

if __name__ == "__main__":
    main()
