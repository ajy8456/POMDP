import io
import torch as th


def save_checkpoint(msg, filename, epoch, val, model, optimizer, scheduler):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'top_eval': val
    }

    th.save(state, filename)
    print(msg)


def load_checkpoint(config, filename, model, optimizer, scheduler):
    # with open(filename, 'rb') as f:
    #     buffer = io.BytesIO(f.read())
    # checkpoint = th.load(buffer, map_location=config.device)
    checkpoint = th.load(filename, map_location=config.device)
    start_epoch = checkpoint['epoch']
    val = checkpoint['top_eval']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return start_epoch, val, model, optimizer, scheduler