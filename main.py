import argparse
import RABAS

def main(input_path):
    rabas = RABAS.RABAS(
        model_id="google/gemma-3-27b-it",
        data_path=input_path,
        config_path="metrics.json",
        metrics=["faithfulness"]
    )

    rabas.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lanzar demo con base de datos")
    parser.add_argument("input_path", help="Path al fichero de configuracion")

    args = parser.parse_args()
    main(args.input_path)