import os
import glob
import time
from dataclasses import dataclass, replace
from simple_parsing import Serializable
from typing import List
import pickle
import torch as th
import numpy as np
from tensorboardX import SummaryWriter

from load import get_loader
from model import GPT2, RNN, LSTM, CVAE, ValueNet
from loss import RegressionLossPolicy, RegressionLossValue, ELBOLoss
from trainer import Trainer
from evaluator import Evaluator
from saver import save_checkpoint, load_checkpoint
from utils import ModelAsTuple, CosineAnnealingWarmUpRestarts, log_gradients


@dataclass
class Settings(Serializable):
    # Dataset
    path: str = 'Learning/dataset'
    # data_type: str = 'mcts' # 'mcts' or 'success'
    data_type: str = 'success' # 'mcts' or 'success'
    # data_type_1: str = 'success' # 'mcts' or 'success'
    # data_type_2: str = 'mcts' # 'mcts' or 'success'
    randomize: bool = True
    filter: float = 51
    train_file: str = 'sim_success_randomize_1' # folder name
    # train_file_1: str = 'sim_success_exp_const_30_std0.5'
    # train_file_2: str = 'mcts_1_exp_const_30_std0.5'
    test_file: str = 'sim_success_randomize_1'
    batch_size: int = 4096 # 100steps/epoch
    shuffle: bool = True # for using Sampler, it should be False
    use_sampler: bool = False
    max_len: int = 100
    seq_len: int = 31
    # |TODO| modify to automatically change
    dim_observation: int = 2
    dim_action: int = 2
    dim_state: int = 2
    dim_reward: int = 1

    # Architecture
    model: str = 'CVAE' # GPT or RNN or LSTM or CVAE or ValueNet or PolicyValueNet
    optimizer: str = 'AdamW' # AdamW or AdamWR

    dim_embed: int = 16
    dim_hidden: int = 16
    # dim_embed: int = 32
    # dim_hidden: int = 32
    # dim_embed: int = 64
    # dim_hidden: int = 64
    # dim_embed: int = 128
    # dim_hidden: int = 128

    # for GPT
    dim_head: int = 16
    num_heads: int = 1
    dim_ffn: int = 16 * 4
    num_layers: int = 3
    # dim_head: int = 32
    # num_heads: int = 1
    # dim_ffn: int = 32 * 4
    # num_layers: int = 3
    # dim_head: int = 64
    # num_heads: int = 1
    # dim_ffn: int = 64 * 4
    # num_layers: int = 3
    # dim_head: int = 128
    # num_heads: int = 1
    # dim_ffn: int = 128 * 3
    # num_layers: int = 3

    # for CVAE
    latent_size: int = 16
    dim_condition: int = 16
    # latent_size: int = 32
    # dim_condition: int = 32
    # latent_size: int = 64
    # dim_condition: int = 64
    # latent_size: int = 128
    # dim_condition: int = 128
    encoder_layer_sizes = [dim_embed, dim_embed + dim_condition, latent_size]
    decoder_layer_sizes = [latent_size, latent_size + dim_condition, dim_action]
    # encoder_layer_sizes = [dim_embed, latent_size]
    # decoder_layer_sizes = [latent_size, dim_action]

    train_pos_en: bool = False
    use_reward: bool = True
    use_mask_padding: bool = True
    coefficient_loss: float = 1e-3

    dropout: float = 0.1
    action_tanh: bool = False

    # Training
    device: str = 'cuda' if th.cuda.is_available() else 'cpu'
    # resume: str = None # checkpoint file name for resuming
    resume: str = 'ckpt_epoch_280.pth' # checkpoint file name for resuming
    pre_trained: str = None
    # pre_trained: str = '3.15_CVAE_dim16_randomize_1/best.pth'
    # pre_trained: str = '3.15_CVAE_mcts_2_dim16/best.pth'
    # pre_trained: str = '3.11_CVAE_mcts_1_dim16/best.pth'
    # pre_trained: str = '3.5_CVAE_sim_mcts_2_dim16/best.pth'
    # pre_trained: str = '2.27_CVAE_sim_mcts_1_dim16/best.pth'
    # pre_trained: str = '2.8_CVAE_sim_dim16/best.pth'
    # pre_trained: str = '12.7_CVAE_mcts2/best.pth' # checkpoint file name for pre-trained model
    # pre_trained: str = '11.23_CVAE_randomized/best.pth' # checkpoint file name for pre-trained model
    # pre_trained: str = '11.29_CVAE_mcts1_filtered/best.pth' # checkpoint file name for pre-trained model
    # pre_trained: str = '12.27_CVAE_sim_huge/best.pth' # checkpoint file name for pre-trained model
    # pre_trained: str = '12.27_CVAE_sim_huge_x/best.pth' # checkpoint file name for pre-trained model
    # |NOTE| Large # of epochs by default, Such that the tranining would *generally* terminate due to `train_steps`.
    epochs: int = 1000

    # Learning rate
    # |NOTE| using small learning rate, in order to apply warm up
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_step: int = int(1e3)
    # For cosine annealing
    T_0: int = int(1e4)
    T_mult: int = 1
    lr_max: float = 0.01
    lr_mult: float = 0.5

    # Logging
    exp_dir: str = 'Learning/exp'
    model_name: str = '4.17_CVAE'
    # model_name: str = '4.17_ValueNet'

    print_freq: int = 100 # per train_steps
    train_eval_freq: int = 100 # per train_steps
    test_eval_freq: int = 10 # per epochs
    save_freq: int = 10 # per epochs

    log_para: bool = False
    log_grad: bool = False
    eff_grad: bool = False
    print_num_para: bool = True
    print_in_out: bool = False


def main():
    config = Settings()
    # |TODO| go to Setting()
    train_filename = config.train_file
    # train_filename_1 = config.train_file_1
    # train_filename_2 = config.train_file_2
    test_filename = config.test_file
    dataset_path = os.path.join(os.getcwd(), config.path)
    
    if not os.path.exists(config.exp_dir):
        os.mkdir(config.exp_dir)
    model_dir = os.path.join(config.exp_dir, config.model_name)

    logger = SummaryWriter(model_dir)

    if config.data_type == 'success':
        # with open(os.path.join(dataset_path, train_filename), 'rb') as f:
        #     train_dataset = pickle.load(f)
        # with open(os.path.join(dataset_path, test_filename), 'rb') as f:
        #     test_dataset = pickle.load(f)

        dataset = glob.glob(f'{dataset_path}/{train_filename}/*.pickle')
        # test_dataset = glob.glob(f'{dataset_path}/{test_filename}/*.pickle')
        train_dataset = dataset[:1500000]
        test_dataset = dataset[-200000:]
        # train_dataset = dataset[:60000]
        # test_dataset = dataset[-10000:]

        print('#trajectories of train_dataset:', len(train_dataset))
        print('#trajectories of test_dataset:', len(test_dataset))
    
    elif config.data_type == 'mcts':
        dataset = glob.glob(f'{dataset_path}/{train_filename}/*.pickle')
        train_dataset = dataset[:-30000]
        test_dataset = dataset[-30000:]

        # train_dataset = glob.glob(f'{dataset_path}/{train_filename}/*.pickle')
        # test_dataset = glob.glob(f'{dataset_path}/{test_filename}/*.pickle')

        if config.filter:
            filtered_data_train = []
            filtered_data_test = []
            total_reward_filt = []
            total_reward_not_filt = []
            avg_total_reward_not_filt = 0
            avg_total_reward_filt = 0
            for data in train_dataset:
                with open(data, 'rb') as f:
                    traj = pickle.load(f)
                    avg_total_reward_not_filt += traj[-1]
                    total_reward_not_filt.append(traj[-1])
                    if traj[-1] > config.filter:
                        filtered_data_train.append(data)
                        avg_total_reward_filt += traj[-1]
                        total_reward_filt.append(traj[-1])

            for data in test_dataset:
                with open(data, 'rb') as f:
                    traj = pickle.load(f)
                    if traj[-1] > config.filter:
                        filtered_data_test.append(data)
                        
            total_reward_not_filt_std = np.std(np.asarray(total_reward_not_filt))
            total_reward_filt_std = np.std(np.asarray(total_reward_filt))
            print('Average of total reward(not filtered):', avg_total_reward_not_filt/len(train_dataset))
            print('std of total reward(not filtered):', total_reward_not_filt_std)
            print('Average of total reward(filtered):', avg_total_reward_filt/len(filtered_data_train))
            print('std of total reward(filtered):', total_reward_filt_std)
            
            train_dataset = filtered_data_train
            test_dataset = filtered_data_test
    
        print('#trajectories of train_dataset:', len(train_dataset))
        print('#trajectories of test_dataset:', len(test_dataset))



    # # For mixed dataset   
    # train_dataset_1 = glob.glob(f'{dataset_path}/{train_filename_1}/*.pickle')
       
    # dataset_2 = glob.glob(f'{dataset_path}/{train_filename_2}/*.pickle')
    # train_dataset_2 = dataset_2[:100000]
    # test_dataset = dataset_2[100000:]

    # if config.filter:
    #     filtered_data_train = []
    #     filtered_data_test = []
    #     total_reward_filt = []
    #     total_reward_not_filt = []
    #     avg_total_reward_not_filt = 0
    #     avg_total_reward_filt = 0
    #     for data in train_dataset_2:
    #         with open(data, 'rb') as f:
    #             traj = pickle.load(f)
    #             avg_total_reward_not_filt += traj[-1]
    #             total_reward_not_filt.append(traj[-1])
    #             if traj[-1] > config.filter:
    #                 filtered_data_train.append(data)
    #                 avg_total_reward_filt += traj[-1]
    #                 total_reward_filt.append(traj[-1])

    #     for data in test_dataset:
    #         with open(data, 'rb') as f:
    #             traj = pickle.load(f)
    #             if traj[-1] > config.filter:
    #                 filtered_data_test.append(data)
                    
    #     total_reward_not_filt_std = np.std(np.asarray(total_reward_not_filt))
    #     total_reward_filt_std = np.std(np.asarray(total_reward_filt))
    #     print('Average of total reward(not filtered):', avg_total_reward_not_filt/len(train_dataset_2))
    #     print('std of total reward(not filtered):', total_reward_not_filt_std)
    #     print('Average of total reward(filtered):', avg_total_reward_filt/len(filtered_data_train))
    #     print('std of total reward(filtered):', total_reward_filt_std)
        
    #     train_dataset = train_dataset_1 + filtered_data_train
    #     test_dataset = filtered_data_test
    
    # print('#trajectories of train_dataset:', len(train_dataset))
    # print('#trajectories of test_dataset:', len(test_dataset))



    # generate dataloader
    train_loader = get_loader(config, train_dataset)
    test_loader = get_loader(config, test_dataset)

    # model
    device = th.device(config.device)
    if config.model == 'GPT':
        model = GPT2(config).to(device)
    elif config.model == 'RNN':
        model = RNN(config).to(device)
    elif config.model == 'LSTM':
        model = LSTM(config).to(device)
    elif config.model == 'CVAE' or config.model == 'PolicyValueNet':
        model = CVAE(config).to(device)
    elif config.model == 'ValueNet':
        model = ValueNet(config).to(device)
    else:
        raise Exception(f'"{config.model}" is not support!! You should select "GPT", "RNN", "LSTM", "CVAE", "ValueNet", or "PolicyValueNet.')

    # optimizer
    optimizer = th.optim.AdamW(model.parameters(),
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    
    # learning rate scheduler
    if config.optimizer == 'AdamW':
        scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1)/config.warmup_step, 1))
    elif config.optimizer == 'AdamWR':
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer=optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_max=config.lr_max,
            T_up=config.warmup_step,
            gamma=config.lr_mult
        )
    else:
        raise Exception(f'"{config.optimizer}" is not support!! You should select "AdamW" or "AdamWR".')

    # Metric
    # |TODO| implement Chamfer distance
    if config.model == 'CVAE':
        loss_fn = ELBOLoss(config)
        eval_fn = ELBOLoss(config)
    elif config.model == 'ValueNet':
        loss_fn = RegressionLossValue(config)
        eval_fn = RegressionLossValue(config)
    elif config.model == 'PolicyValueNet':
        loss_fn = None
        eval_fn = None
    else:
        loss_fn = RegressionLossPolicy(config)
        eval_fn = RegressionLossPolicy(config)

    # Trainer & Evaluator
    trainer = Trainer(config=config,
                      loader=train_loader,
                      model=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      loss_fn=loss_fn,
                      eval_fn=eval_fn)
    evaluator = Evaluator(config=config,
                          loader=test_loader,
                          model=model,
                          eval_fn=eval_fn)

    # save configuration
    config.save(model_dir + '/config.yaml')
    # Logging model graph
    dummy = next(iter(test_loader))
    for k in dummy:
        dummy[k].to(device).detach()
    logger.add_graph(ModelAsTuple(config, model), dummy)

    start_epoch = 1
    best_error = 10000.

    # load checkpoint for resuming
    if config.resume is not None:
        filename = os.path.join(model_dir, config.resume)
        if os.path.isfile(filename):
            start_epoch, best_error, model, optimizer, scheduler = load_checkpoint(config, filename, model, optimizer, scheduler)
            start_epoch += 1
            print("Loaded checkpoint '{}' (epoch {})".format(config.resume, start_epoch))
        else:
            raise Exception("No checkpoint found at '{}'".format(config.resume))

    # load checkpoint for pre-trained
    if config.pre_trained is not None:
        pre_trained_path = os.path.join(config.exp_dir, config.pre_trained)
        if os.path.isfile(pre_trained_path):
            start_epoch, best_error, model, optimizer, scheduler = load_checkpoint(config, pre_trained_path, model, optimizer, scheduler)
            start_epoch = 1
            print("Loaded checkpoint '{}'".format(config.pre_trained))
        else:
            raise Exception("No checkpoint found at '{}'".format(config.resume))

    for epoch in range(start_epoch, config.epochs+1):
        print(f'===== Start {epoch} epoch =====')
        
        # Training one epoch
        print("Training...")
        train_loss, train_val = trainer.train(epoch)

        # Logging
        if config.model == 'CVAE':
            logger.add_scalar('Loss(total)/train', train_loss['total'], epoch)
            logger.add_scalar('Loss(Reconstruction)/train', train_loss['Recon'], epoch)
            logger.add_scalar('Loss(KL_divergence)/train', train_loss['KL_div'], epoch)
        elif config.model == 'ValueNet':
            logger.add_scalar('Loss/train', train_loss['total'], epoch)
        elif config.model == 'PolicyValueNet':
            logger.add_scalar('Loss(total)/train', train_loss['total'], epoch)
            logger.add_scalar('Loss(action)/train', train_loss['action'], epoch)
            logger.add_scalar('Loss(accumulated reward)/train', train_loss['accumulated_reward'], epoch)
            # logger.add_scalar('Eval(action)/train', train_val['action'], epoch)
        else:
            logger.add_scalar('Loss(total)/train', train_loss['total'], epoch)
            logger.add_scalar('Loss(action)/train', train_loss['action'], epoch)
            # if config.use_reward:
            #     logger.add_scalar('Loss(reward)/train', train_loss['reward'], epoch)

            # logger.add_scalar('Eval(action)/train', train_val['action'], epoch)
            # if config.use_reward:
            #     logger.add_scalar('Eval(reward)/train', train_val['reward'], epoch)

        # |FIXME| debug for eff_grad: "RuntimeError: Boolean value of Tensor with more than one value is ambiguous"
        log_gradients(model, logger, epoch, log_grad=config.log_grad, log_param=config.log_para, eff_grad=config.eff_grad, print_num_para=config.print_num_para)

        # evaluating
        if epoch % config.test_eval_freq == 0:
            print("Validating...")
            test_val = evaluator.eval(epoch)

            # save the best model
            # |TODO| change 'action' to 'total' @ trainer.py & evaluator.py -> merge 'CVAE' & others
            if config.model == 'CVAE' or config.model == 'ValueNet' or config.model == 'PolicyValueNet':
                if test_val['total'] < best_error:
                    best_error = test_val['total']

                    save_checkpoint('Saving the best model!',
                                    os.path.join(model_dir, 'best.pth'),
                                    epoch, 
                                    best_error, 
                                    model, 
                                    optimizer, 
                                    scheduler
                                    )
            else:
                if test_val['action'] < best_error:
                    best_error = test_val['action']

                    save_checkpoint('Saving the best model!',
                                    os.path.join(model_dir, 'best.pth'),
                                    epoch, 
                                    best_error, 
                                    model, 
                                    optimizer, 
                                    scheduler
                                    )
            
            # Logging
            if config.model == 'CVAE':
                logger.add_scalar('Eval(total)/test', test_val['total'], epoch)
                logger.add_scalar('Eval(Reconstruction)/test', test_val['Recon'], epoch)
                logger.add_scalar('Eval(KL_divergence)/test', test_val['KL_div'], epoch)
            elif config.model == 'ValueNet':
                logger.add_scalar('Eval/test', test_val['total'], epoch)
            elif config.model == 'PolicyValueNet':
                logger.add_scalar('Eval(total)/test', test_val['total'], epoch)
                logger.add_scalar('Eval(action)/test', test_val['action'], epoch)
                logger.add_scalar('Eval(accumulated reward)/test', test_val['accumulated_reward'], epoch)
            else:
                logger.add_scalar('Eval(action)/test', test_val['action'], epoch)
                # if config.use_reward:
                #     logger.add_scalar('Eval(reward)/test', test_val['reward'], epoch)
        
        # save the model
        if epoch % config.save_freq == 0:
            save_checkpoint('Saving...', 
                            os.path.join(model_dir, f'ckpt_epoch_{epoch}.pth'), 
                            epoch, 
                            best_error, 
                            model, 
                            optimizer, 
                            scheduler
                            )

        print(f'===== End {epoch} epoch =====')


if __name__ == '__main__':
    total_time_start = time.time()
    main()
    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    print("Total Time:", total_time)