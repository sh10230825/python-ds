# Training loop for one epoch

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for src, trg in iterator:
        # Ensure src and trg are on the same device as the model
        src = src.to(model.encoder.embedding.weight.device)
        trg = trg.to(model.encoder.embedding.weight.device)

        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
