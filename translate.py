# Translate a single Chinese sentence into English

import torch

def translate_sentence(sentence, model, src_vocab, trg_vocab, device, max_len=30):
    model.eval()
    
    # Tokenize input and convert to tensor
    tokens = ['<sos>'] + list(sentence.strip()) + ['<eos>']
    src_ids = [src_vocab.stoi.get(t, src_vocab.stoi['<unk>']) for t in tokens]
    src_tensor = torch.tensor(src_ids).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden = model.encoder(src_tensor)

    trg_indexes = [trg_vocab.stoi['<sos>']]

    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden)
            pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab.stoi['<eos>']:
            break

    trg_tokens = trg_vocab.decode(trg_indexes)
    return ' '.join(trg_tokens[1:-1])
