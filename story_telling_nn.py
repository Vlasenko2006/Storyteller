import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from Tokenizer import TransformerTokenizer
from TextDataset import TransformerTextDataset, transformer_collate_fn
from LabelSmoothingLoss import LabelSmoothingLoss
from my_trainer import my_trainer
from seeders import seeders
from final_text import final_text
from model import MiniBertForNextWordPrediction, MiniTransformerWithEncoderDecoder 
import yaml 
from clean_text import clean_text

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main_worker(rank, world_size, config, ddp_enabled):
    if ddp_enabled:
        # Flexible: get local_rank from SLURM_LOCALID, default to 0
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        print(f"[DDP SETUP] rank={rank}, world_size={world_size}, local_rank={local_rank}, hostname={os.uname().nodename}", flush=True)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    DEBUG_FIRST_N_BATCHES = config["DEBUG_FIRST_N_BATCHES"] 

    # Unpack config
    model_cfg = config['model']
    training_cfg = config['training']
    predictor_cfg = config['predictor']
    path = config['path']
    predicted_steps = config['predicted_steps']
    checkpoint_path = config['checkpoint_path']

    use_adamw = config['use_adamw']
    alternate_costs = config['alternate_costs']
    train_the_model = config['train_the_model']
    load_model = config['load_model']

    # Load and preprocess text corpus (do this only in rank 0 to avoid duplicated print)
    if rank == 0 or not ddp_enabled:
        with open(f"{path}text", "rb") as fp:
            corpus = pickle.load(fp)
        with open(f"{path}preprocessed_qa_dataset_plain.pkl", "rb") as fp:
            qa = pickle.load(fp)
        print(type(corpus))
        if isinstance(corpus, list):
            print(type(corpus[0]))  # e.g., str
    else:
        corpus = None
        qa = None

    # Broadcast data to all ranks in DDP
    if ddp_enabled:
        data_list = [corpus, qa]
        dist.broadcast_object_list(data_list, src=0)
        corpus, qa = data_list

    corpus = clean_text(corpus)
    corpus = corpus + qa

    tokenizer = TransformerTokenizer(
        max_length=model_cfg['max_length'],
        pretrained_model_path_or_name=config['pretrained_model_path_or_name']
    )
    total_words = tokenizer.vocab_size

    dataset = TransformerTextDataset(
        corpus,
        tokenizer,
        max_length=model_cfg['max_length'],
        predict_steps=predicted_steps,
        for_encoder_decoder=(config["model_type"] == "BART")
    )

    if ddp_enabled:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=transformer_collate_fn,
        drop_last=True
    )

    # Model instantiation
    if config["model_type"] == "BERT":
        model = MiniBertForNextWordPrediction(vocab_size=total_words, **model_cfg)
    elif config["model_type"] == "BART":
        model = MiniTransformerWithEncoderDecoder(vocab_size=total_words, **model_cfg)
    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}")

    model.to(device)

    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion_ls = LabelSmoothingLoss(classes=total_words, smoothing=training_cfg['smoothing'])
    criterion_ce = nn.CrossEntropyLoss()

    if use_adamw:
        optimizer_ce = torch.optim.AdamW(model.parameters(), lr=training_cfg['lr_ce'])
        optimizer_ls = torch.optim.AdamW(model.parameters(), lr=training_cfg['lr_ls'])
        optimizer = optimizer_ce
    else:
        optimizer_ce = torch.optim.Adam(model.parameters(), lr=training_cfg['lr_ce'])
        optimizer_ls = torch.optim.Adam(model.parameters(), lr=training_cfg['lr_ls'])
        optimizer = optimizer_ce

    if load_model:
        checkpoint_path_load = os.path.join(checkpoint_path, f"story_telling-{config['predicted_steps']}_ep_{config['load_epoch']}.pth")
        if rank == 0 or not ddp_enabled:
            print("Loading checkpoint: ", checkpoint_path)
        map_location = {"cuda:%d" % 0: f"cuda:{local_rank}"} if ddp_enabled else device
        checkpoint = torch.load(checkpoint_path_load, map_location=map_location)
        if ddp_enabled:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer_ce = optimizer
        start_epoch = checkpoint.get("epoch", 0) + 1
    else:
        start_epoch = 0

    if train_the_model:
        my_trainer(
            nepochs=training_cfg['nepochs'],
            path=checkpoint_path,
            alternate_costs=alternate_costs,
            criterion_ce=criterion_ce,
            criterion_ls=criterion_ls,
            optimizer_ce=optimizer_ce,
            optimizer_ls=optimizer_ls,
            model=model,
            train_loader=train_loader,
            tokenizer=tokenizer,
            device=device,
            predicted_steps=predicted_steps,
            validate_after_nepochs=predictor_cfg['validate_after_nepochs'],
            seeders=seeders,
            start_epoch=start_epoch,
            model_type=config["model_type"],
            DEBUG_FIRST_N_BATCHES=DEBUG_FIRST_N_BATCHES,
            ddp_enabled=ddp_enabled,
            rank=rank
        )
    else:
        assert load_model, "To validate only you must set load_model to 'True' and specify the loading checkpoint!"
        if rank == 0 or not ddp_enabled:
            for index, seeder in enumerate(seeders, start=1):
                text = final_text(
                    seeder,
                    model.module if ddp_enabled else model,
                    tokenizer,
                    num_words=predictor_cfg['num_words'],
                    device=device
                )
                print(f"{index}: {text[0]}")
    if ddp_enabled:
        dist.destroy_process_group()

def parse_config():
    yaml_path = "config/conf.yaml"
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel (multi-GPU)")
    args = parser.parse_args()

    config = parse_config()
    ddp_enabled = args.ddp

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    print(f"STARTUP: SLURM_PROCID={os.environ.get('SLURM_PROCID')}, SLURM_NTASKS={os.environ.get('SLURM_NTASKS')}, SLURM_LOCALID={os.environ.get('SLURM_LOCALID')}, HOSTNAME={os.uname().nodename}, ddp_enabled={ddp_enabled}, rank={rank}, world_size={world_size}", flush=True)

    if ddp_enabled:
        main_worker(rank, world_size, config, ddp_enabled)
    else:
        main_worker(rank=0, world_size=1, config=config, ddp_enabled=False)
