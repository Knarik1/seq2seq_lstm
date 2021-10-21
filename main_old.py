import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import argparse
import math
import time
import logging
from aim import Session

import torch
import torch.optim as optim
import torch.utils.data

from utilities import (create_folder, get_filename, create_logging,
                       StatisticsContainer)
from utils.data_generator_old import MaestroDataset, Augmentor, Sampler, collate_fn
from pytorch_utils import move_data_to_device
from losses import get_loss_func
from utils import config

aim_recording = False
# Aim - track hyper parameters
if aim_recording:
    sess = Session(experiment='Continue_pretraining')


def train(args):
    """Train a piano transcription system.

    Args:
      workspace: str, directory of your workspace
      model_type: str, e.g. 'Regressonset_regressoffset_frame_velocity_CRNN'
      loss_type: str, e.g. 'regress_onset_offset_frame_velocity_bce'
      checkpoint: str, e.g. 'note_model.pth'
      augmentation: str, e.g. 'none'
      batch_size: int
      learning_rate: float
      reduce_iteration: int
      resume_iteration: int
      early_stop: int
      device: 'cuda' | 'cpu'
      mini_data: bool
    """

    # Arugments & parameters
    workspace = args.workspace
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    max_note_shift = args.max_note_shift
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reduce_iteration = args.reduce_iteration
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    NOTE_CHECKPOINT_PATH = args.checkpoint


    mini_data = args.mini_data
    filename = args.filename

    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    hop_seconds = config.hop_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    num_workers = 1
    if aim_recording:
        sess.set_params({
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'Notes': 'no freezing layer, reduced dataset to balance octaves 10 sec with  continue training no quant '
        },
            name='hparams')


    # Loss function
    loss_func = get_loss_func(loss_type)

    # Paths
    hdf5s_dir = os.path.join(workspace, 'hdf5s')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift),
        'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift), 
        'batch_size={}'.format(batch_size), 'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift), 
        'batch_size={}'.format(batch_size))
    create_folder(logs_dir)

    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'
    
    # Model
    model_type = 'Regress_onset_offset_frame_velocity_CRNN'
    Model = eval(model_type)
    model = Model(frames_per_second=frames_per_second, classes_num=classes_num)
    print("frames_per_second", frames_per_second)
    print("classes_num", classes_num)

    if augmentation == 'none':
        augmentor = None
    elif augmentation == 'aug':
        augmentor = Augmentor()
    else:
        raise Exception('Incorrect argumentation!')

    # Dataset
    print("hdf5s_dir", "---", hdf5s_dir)
    train_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir, 
        segment_seconds=segment_seconds, frames_per_second=frames_per_second, 
        max_note_shift=max_note_shift, augmentor=augmentor)

    # print("evaluate_dataset11111111111111111111")
    # evaluate_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir,
    #     segment_seconds=segment_seconds, frames_per_second=frames_per_second,
    #     max_note_shift=0)

    # Sampler for training
    train_sampler = Sampler(hdf5s_dir=hdf5s_dir, split='train', 
        segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    # # Sampler for evaluation
    # evaluate_train_sampler = TestSampler(hdf5s_dir=hdf5s_dir,
    #     split='train', segment_seconds=segment_seconds, hop_seconds=hop_seconds,
    #     batch_size=batch_size, mini_data=mini_data)
    #
    # evaluate_validate_sampler = TestSampler(hdf5s_dir=hdf5s_dir,
    #     split='validation', segment_seconds=segment_seconds, hop_seconds=hop_seconds,
    #     batch_size=batch_size, mini_data=mini_data)
    #
    # evaluate_test_sampler = TestSampler(hdf5s_dir=hdf5s_dir,
    #     split='test', segment_seconds=segment_seconds, hop_seconds=hop_seconds,
    #     batch_size=batch_size, mini_data=mini_data)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    # print("train_loader= ",train_loader.keys())


    # evaluate_train_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset,
    #     batch_sampler=evaluate_train_sampler, collate_fn=collate_fn,
    #     num_workers=num_workers, pin_memory=True)
    #
    # validate_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset,
    #     batch_sampler=evaluate_validate_sampler, collate_fn=collate_fn,
    #     num_workers=num_workers, pin_memory=True)
    #
    # test_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset,
    #     batch_sampler=evaluate_test_sampler, collate_fn=collate_fn,
    #     num_workers=num_workers, pin_memory=True)

    # Evaluator
    evaluator = SegmentEvaluator(model, batch_size)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    # Resume training
    if resume_iteration > 0:
        # Node model path here. Note model is the first model of original model
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', NOTE_CHECKPOINT_PATH)

        # logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint)

        # Freezing all layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Unfreeze layers
        # unfreeze_layers = [model.frame_model, model.reg_onset_model, model.reg_offset_model, model.velocity_model]

        # for layer_params in unfreeze_layers:
        #     for param in layer_params.fc.parameters():
        #         param.requires_grad = True

            # init last FC layers randomly
            # init_layer(layer_params.fc)

        iteration = resume_iteration

    else:
        iteration = 0


    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)


    if 'cuda' in str(device):
        model.to(device)

    train_bgn_time = time.time()

    epoch = 0
    for batch_data_dict in train_loader:
        # Evaluation
        if iteration % 5000 == 0:# and iteration > 0:
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()

            # Logging

            # evaluate_train_statistics = evaluator.evaluate(evaluate_train_loader)
            # validate_statistics = evaluator.evaluate(validate_loader)
            # test_statistics = evaluator.evaluate(test_loader)
            #
            # logging.info('    Train statistics: {}'.format(evaluate_train_statistics))
            # logging.info('    Validation statistics: {}'.format(validate_statistics))
            # logging.info('    Test statistics: {}'.format(test_statistics))
            #
            # statistics_container.append(iteration, evaluate_train_statistics, data_type='train')
            # statistics_container.append(iteration, validate_statistics, data_type='validation')
            # statistics_container.append(iteration, test_statistics, data_type='test')
            # statistics_container.dump()
            #
            # train_time = train_fin_time - train_bgn_time
            # validate_time = time.time() - train_fin_time
            #
            # logging.info(
            #     'Train time: {:.3f} s, validate time: {:.3f} s'
            #     ''.format(train_time, validate_time))
            #
            # train_bgn_time = time.time()

        # Save model
        # if iteration % 20000 == 0:
        #     checkpoint = {
        #         'iteration': iteration,
        #         'model': model.module.state_dict(),
        #         'sampler': train_sampler.state_dict()}
        #
        #     checkpoint_path = os.path.join(
        #         workspace, 'checkpoints', 'iter_{}.pth'.format(iteration))
        #
        #     torch.save(checkpoint, checkpoint_path)
        #     logging.info('Model saved to {}'.format(checkpoint_path))
        #
        # # Reduce learning rate
        # if iteration % reduce_iteration == 0 and iteration > 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.9

        # Move data to device
        print("batch_data_dict.keys()",batch_data_dict.keys())
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        print("=================================================data")
        print(batch_data_dict[0])
        exit()
        model.train()
        batch_output_dict = model(batch_data_dict['waveform'])

        loss = loss_func(model, batch_output_dict, batch_data_dict)

        if aim_recording:
            sess.track(loss, name='train_loss', epoch=iteration)

        print(iteration, '/', early_stop, 'loss = ', round(loss.item(), 4))
        if  math.isnan(loss.item()):
            exit()
            return
        # Backward
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        # Save at every n epoch
        save_every_epoch = 1

        if train_loader.batch_sampler.epoch % save_every_epoch == 0 and train_loader.batch_sampler.epoch > epoch:
            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations_fixed.pth'.format(iteration))

            torch.save(model.module.state_dict(), checkpoint_path)
            print("model saved at", checkpoint_path)
            if aim_recording:
                sess.set_params({'model_name': checkpoint_path})

            epoch += 1
        
        # Stop learning
        if iteration == early_stop:

            # Uncomment to save model in original form

            # checkpoint = {
            #     'iteration': iteration,
            #     'model': model.module.state_dict(),
            #     'sampler': train_sampler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations_fixed.pth'.format(iteration))

            torch.save(model.module.state_dict(), checkpoint_path)
            print("model saved at", checkpoint_path)
            break
        print("=============================================================================")
        iteration += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--checkpoint', type=str, required=False)
    parser_train.add_argument('--augmentation', type=str, required=True, choices=['none', 'aug'])

    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--reduce_iteration', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int, required=True)
    parser_train.add_argument('--early_stop', type=int, required=True)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        print("-----------------------djsdkjhskjh")
        train(args)

    else:
        raise Exception('Error argument!')
