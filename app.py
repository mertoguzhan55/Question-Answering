import torch
from app.config import Configs
from app.logger import Logger
import os
from app.train import LlamaTrainer
from app.llama_inference import LlamaInference
from app.fast_api import FastAPIServer



def main(args, configs):

    logger = Logger(**configs["logger"])

    logger.debug("############ MODEL TRAINING CONFIGURATIONS ############")
    logger.debug(configs)

    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    if args.train:
        trainer = LlamaTrainer(**configs["LlamaTrain"], logger=logger)
        trainer.train()
    elif args.infer:
        infer = LlamaInference(**configs["LlamaInference"], logger = logger)
        print(infer.generate_text())
    elif args.fastapi:
        infer = LlamaInference(**configs["LlamaInference"], logger = logger)
        fastapi = FastAPIServer(**configs["FastapiServer"], model = infer, logger = logger)
        fastapi.run()
    
    

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", type=str)
    parser.add_argument("-t", "--train", action='store_true')
    parser.add_argument("-i", "--infer", action='store_true')
    parser.add_argument("-f", "--fastapi", action='store_true')
    parser.add_argument("--test", action= "store_true")


    args = parser.parse_args()

    configs = Configs().load(config_name=args.environment)

    main(args, configs)
