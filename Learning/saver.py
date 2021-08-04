import torch as th


def save_checkpoint(epoch, model, optimizer, val, filename, msg):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'top_eval': val
    }
    th.save(state, filename)
    print(msg)


def load_checkpoint(model, optimizer, filename):
    checkpoint = th.load(filename, map_location='cuda:0')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return start_epoch, model, optimizer