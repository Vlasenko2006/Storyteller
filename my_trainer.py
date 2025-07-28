import torch
from tqdm import tqdm
from multi_word_loss import multi_word_loss
from final_text import final_text

def limited_batches_loader(loader, n_batches):
    """Yield only the first n_batches from loader, then stop."""
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        yield batch

def my_trainer(
    nepochs,
    path,
    alternate_costs, 
    criterion_ce, 
    criterion_ls, 
    optimizer_ce, 
    optimizer_ls, 
    model,
    train_loader,
    tokenizer,
    device='cpu', 
    predicted_steps=1,
    validate_after_nepochs=1,
    seeders=["NaN"],
    start_epoch=0,
    grad_accum_steps=1,
    model_type="BERT",
    DEBUG_FIRST_N_BATCHES = None,
    ddp_enabled=False,
    rank=0
):
    import torch.distributed as dist

    text = ["NaN"]
    for epoch in range(start_epoch, nepochs):
        if DEBUG_FIRST_N_BATCHES:
            epoch_train_loader = limited_batches_loader(train_loader, DEBUG_FIRST_N_BATCHES)
        else:
            epoch_train_loader = train_loader

        # If using DistributedSampler, set epoch for shuffling

        # If using DistributedSampler, set epoch for shuffling
        #if ddp_enabled and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
        #    train_loader.sampler.set_epoch(epoch)

        if alternate_costs:
            if epoch % 2 == 1: 
                criterion = criterion_ce 
                optimizer = optimizer_ce
            else:
                criterion = criterion_ls
                optimizer = optimizer_ls
        else:
            criterion = criterion_ce
            optimizer = optimizer_ce

        model.train()
        epoch_loss = 0
        batch_count = 0

        # Only rank 0 shows the progress bar
        iter_obj = tqdm(epoch_train_loader, desc=f"Epoch {epoch + 1}") if (not ddp_enabled or rank == 0) else epoch_train_loader

        optimizer.zero_grad()
        for step, batch in enumerate(iter_obj):
            if ddp_enabled:
                print(f"Rank {rank} sees batch {step}, batch size: {batch['input_ids'].size(0)}")
            if model_type == "BART":  # Encoder-decoder
                src_input_ids = batch["src_input_ids"].to(device)
                src_attention_mask = batch["src_attention_mask"].to(device)
                tgt_input_ids = batch["tgt_input_ids"].to(device)
                tgt_attention_mask = batch["tgt_attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(
                    src_input_ids, tgt_input_ids,
                    src_attention_mask=src_attention_mask,
                    tgt_attention_mask=tgt_attention_mask
                )

                loss = multi_word_loss(logits, labels, criterion)
            else:  # Default: BERT-like
                X = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)

                output = model(X, attention_mask=attention_mask)
                loss = multi_word_loss(output, y, criterion)

            loss = loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * grad_accum_steps
            batch_count += 1

            if (not ddp_enabled or rank == 0) and hasattr(iter_obj, 'set_postfix'):
                iter_obj.set_postfix(loss=loss.item() * grad_accum_steps)

        # Optionally reduce loss across all ranks for correct average
        if ddp_enabled:
            avg_loss_tensor = torch.tensor([epoch_loss], dtype=torch.float32, device=device)
            torch.distributed.all_reduce(avg_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / (batch_count * dist.get_world_size())
        else:
            avg_loss = epoch_loss / batch_count if batch_count > 0 else float('nan')

        if not ddp_enabled or rank == 0:
            print(f"NN2 Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
            # Save checkpoint only on rank 0
            to_save = model.module if ddp_enabled else model
            torch.save(
                {
                    'model_state_dict': to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                },
                f"{path}/story_telling-lr-{predicted_steps}_ep_{epoch}.pth"
            )

        if ((epoch + 1) % validate_after_nepochs == 0) and (not ddp_enabled or rank == 0):
            for index, seeder in enumerate(seeders, start=1):
                text = final_text(
                    seeder,
                    model.module if ddp_enabled else model,
                    tokenizer,
                    num_words=100,
                    device=device,
                    model_type=model_type  
                )
                print(f"{index}: {text[0]}\n")
    return text
