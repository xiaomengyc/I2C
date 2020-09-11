import os
import torch
import pdb

__all__ = ['restore']


def find_lasted_save_checkpoint(restore_dir):
    filelist = os.listdir(restore_dir)
    filelist = [x for x in filelist if os.path.isfile(os.path.join(restore_dir,x)) and x.endswith('.pth.tar')]
    if len(filelist) > 0:
        filelist.sort(key=lambda fn:os.path.getmtime(os.path.join(restore_dir, fn)), reverse=True)
        snapshot = os.path.join(restore_dir, filelist[0])
    else:
        snapshot = None
    return snapshot

def full_restore(args, model, optimizer=None, istrain=True, including_opt=False):
    if os.path.isfile(args.restore_from) and ('.pth' in args.restore_from):
        snapshot = args.restore_from
    else:
        snapshot = find_lasted_save_checkpoint(args.snapshot_dir)

    if os.path.isfile(snapshot):
        print("=> loading checkpoint '{}'".format(snapshot))
        checkpoint = torch.load(snapshot)

        args.current_epoch = checkpoint['epoch'] + 1
        args.global_counter = checkpoint['global_counter'] + 1
        model.load_state_dict(checkpoint["state_dict"])



def restore(args, model, optimizer, istrain=True, including_opt=False):
    if os.path.isfile(args.restore_from) and ('.pth' in args.restore_from):
        snapshot = args.restore_from
    else:
        restore_dir = args.snapshot_dir
        filelist = os.listdir(restore_dir)
        filelist = [x for x in filelist if os.path.isfile(os.path.join(restore_dir,x)) and x.endswith('.pth.tar')]
        if len(filelist) > 0:
            filelist.sort(key=lambda fn:os.path.getmtime(os.path.join(restore_dir, fn)), reverse=True)
            snapshot = os.path.join(restore_dir, filelist[0])
        else:
            snapshot = ''

    if os.path.isfile(snapshot):
        print("=> loading checkpoint '{}'".format(snapshot))
        checkpoint = torch.load(snapshot)
        try:
            if istrain:
                # args.current_epoch = checkpoint['epoch'] + 1
                # args.global_counter = checkpoint['global_counter'] + 1
                if including_opt:
                    optimizer.load_state_dict(checkpoint['optimizer'])
            # model.load_state_dict(checkpoint['state_dict'])
            if args.resume == "True":
                args.current_epoch = checkpoint['epoch'] + 1
                args.global_counter = checkpoint['global_counter'] + 1

            model_dict = model.state_dict()
            model_keys = list(model_dict.keys())

            # invalid_words = ['edge_branch','cls_fc8']
            # model_dict.update({k:v for k,v in checkpoint['state_dict'].items() if all(word not in k for word in invalid_words)})

            new_dict = {k:v for k,v in checkpoint['state_dict'].items() if (k in model_keys) and (v.size() == model_dict[k].size())}
            print('The following parameters cannot be reloaded!:')
            print([k for k in model_keys if k not in list(new_dict.keys())])
            model_dict.update(new_dict)
            model.load_state_dict(model_dict)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(snapshot, checkpoint['epoch']))
        except KeyError:
            print("KeyError")
            _model_load(model, checkpoint)
        except KeyError:
            print("Loading pre-trained values failed.")
            raise
        print("=> loaded checkpoint '{}'".format(snapshot))
    else:
        print("=> no checkpoint found at '{}'".format(snapshot))


def _model_load(model, pretrained_dict):
    model_dict = model.state_dict()

    if list(model_dict.keys())[0].startswith('module.'):
        pretrained_dict = {'module.'+k: v for k, v in pretrained_dict.items()}

    pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in list(model_dict.keys())}
    print("Weights cannot be loaded:")
    print([k for k in list(model_dict.keys()) if k not in list(pretrained_dict.keys())])

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
