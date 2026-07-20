import argparse

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Download and save a Hugging Face tokenizer locally"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Hugging Face model name (e.g. bert-base-uncased, distilbert-base-uncased)",
    )
    parser.add_argument(
        "--tokenizers-path",
        type=str,
        default="./tokenizers",
        help="Base directory where tokenizers will be stored",
    )

    args = parser.parse_args()

    output_dir = f"{args.tokenizers_path}/{args.model_name}"

    print(f"Downloading tokenizer: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Saving tokenizer to: {output_dir}")
    tokenizer.save_pretrained(output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
