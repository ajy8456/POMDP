import torch as th


def save_checkpoint(msg, filename, epoch, val, model, optimizer, scheduler=None):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'top_eval': val
    }
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    
    th.save(state, filename)
    print(msg)


def load_checkpoint(model, optimizer, filename, scheduler=None):
    checkpoint = th.load(filename, map_location='cuda:0')
    start_epoch = checkpoint['epoch']
    val = checkpoint['top_eval']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return start_epoch, val, model, optimizer, scheduler