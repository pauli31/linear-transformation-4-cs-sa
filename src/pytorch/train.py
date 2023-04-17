import logging
import sys
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT
from src.utils.general_utils import format_time, print_time_info

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)

import torch

FAST_DEBUG = False
FAST_DEBUG_PREDICTIONS = FAST_DEBUG
NUM_FAST_DEBUG_ITER = 100



def run_training(args, model, train_data_iter, val_data_iter, loss_fn, optimizer, scheduler,
                 device, val_target_lang_data_iter=None):
    if val_data_iter is not None:
        validation_size = len(val_data_iter)
    num_epochs = args.epoch_num

    print('\ndevice: {0:s}\n'.format(device.type))

    history = defaultdict(list)
    best_accuracy = 0


    total_train_time = 0
    total_train_examples = 0

    total_val_time = 0
    total_val_examples = 0

    t00 = time.time()
    for epoch in range(num_epochs):
        logger.info('')
        logger.info('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
        logger.info('-' * 30)
        t0 = time.time()

        train_acc, train_loss, epoch_examples = train_epoch(args, model, train_data_iter, loss_fn, optimizer, scheduler,
                                            device, epoch)

        train_time_sec = time.time() - t0
        print_time_info(train_time_sec, epoch_examples, "train_epoch", file=args.result_file)
        train_time = format_time(train_time_sec)

        total_train_time += train_time_sec
        total_train_examples += epoch_examples
        logger.info(f'Train loss: {train_loss} accuracy: {train_acc}')

        t0 = time.time()

        if val_data_iter is not None and validation_size > 0:
            val_acc, val_loss, val_examples = eval_model(args, model, val_data_iter, loss_fn, device)
        else:
            val_acc = val_loss = val_examples = 0

        val_time_sec = time.time() - t0
        val_time = format_time(val_time_sec)
        print_time_info(val_time_sec, val_examples, "validation_epoch", file=args.result_file)

        total_val_time += val_time_sec
        total_val_examples += val_examples

        logger.info(f'Val   loss {val_loss} accuracy {val_acc}')

        if val_target_lang_data_iter is not None:
            target_val_acc, target_val_loss, trg_val_examples = eval_model(args, model, val_target_lang_data_iter, loss_fn, device)
            logger.info(f'Target language val   loss {target_val_loss} accuracy {target_val_acc}')
        else:
            target_val_acc = 0
            target_val_loss = 0

        if val_acc > best_accuracy:
            # TODO mozna pouzit tu tridu Metrics, mozna to ukladat podle fmeasure a dat nazev podle configu + do nej dat asi i epochu
            # + a fmeasure
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

        total_time = time.time() - t00
        logger.info(f'Total time for one epoch including validation time: {total_time}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['train_time'].append(train_time)
        history['val_time'].append(val_time)
        history['total_time'].append(total_time)

    print_time_info(total_train_time, total_train_examples, "entire_training", file=args.result_file)
    print_time_info(total_val_time, total_val_examples, "entire_validation", file=args.result_file)

    return history


def train_epoch(args, model, train_data_iter, loss_fn, optimizer,
                scheduler, device, epoch):
    model = model.train()

    losses = []
    correct_predictions = 0
    correct_pred_tmp = 0

    running_loss = 0.0
    n_examples = 0
    total_processed_examples = 0
    # time since epoch started
    t0 = time.time()

    data_loader_len = len(train_data_iter)

    batch_times = []

    # the true number wil be little bit lower, bcs we do not align the data
    total_examples = data_loader_len * args.batch_size

    print_batch_freq = args.print_stat_frequency

    i = 0
    for batch in tqdm(train_data_iter, file=sys.stdout):
        i = i + 1
        t_batch = time.time()
        if FAST_DEBUG is True:
            # only for testing purposes
            if i == NUM_FAST_DEBUG_ITER:
                break

        input_ids = batch.text[0].to(device)
        input_ids_lengths = batch.text[1]
        labels = batch.label.to(device)

        data_size = len(labels)
        n_examples += data_size
        total_processed_examples += data_size

        predictions = model(input_ids, input_ids_lengths)
        loss = loss_fn(predictions, labels)

        _, preds = torch.max(predictions, dim=1)

        tmp = torch.sum(preds == labels)
        correct_pred_tmp += tmp
        correct_predictions += tmp
        if args.data_parallel is True:
            # https://discuss.pytorch.org/t/loss-function-in-multi-gpus-training-pytorch/76765
            # https://discuss.pytorch.org/t/how-to-fix-gathering-dim-0-warning-in-multi-gpu-dataparallel-setting/41733
            # print("loss:" + str(loss))
            # loss_mean = loss.mean()
            # print("Loss mean:" + str(loss_mean))
            loss = loss.mean()

        loss_item = loss.item()
        losses.append(loss_item)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # store the batch time
        batch_times.append(time.time() - t_batch)

        running_loss += loss_item

        # print every n mini-batches
        if i % print_batch_freq == 0 and not i == 0:
            try:
                last_lr = scheduler.get_last_lr()
                last_lr = last_lr[0]
            except Exception as e:
                last_lr = 0
                logger.error("Cannot parse actual learning rate")

            avg_batch_time = np.mean(batch_times)
            eta = (data_loader_len - i) * avg_batch_time

            avg_batch_time = format_time(avg_batch_time)
            eta = format_time(eta)

            elapsed = format_time(time.time() - t0)
            avg_loss = running_loss / print_batch_freq
            acc = correct_pred_tmp.double() / n_examples
            logger.info(
                'Batch: %5d/%-5d avg loss: %.4f  acc: %.3f processed: %d/%d examples,  epoch-time: %s, actual lr:%.12f eta:%s avg-batch-time:%s' %
                (i, len(train_data_iter), avg_loss, acc,
                 total_processed_examples, total_examples, elapsed, last_lr, str(eta), str(avg_batch_time)))

            running_loss = 0.0
            n_examples = 0.0
            correct_pred_tmp = 0.0

    logger.info(f"Number of examples: {total_processed_examples}")
    logger.info(f"Correct predictions: {correct_predictions}")
    return correct_predictions.double() / total_processed_examples, np.mean(losses), total_processed_examples


def eval_model(args, model, data_iterator, loss_fn, device):
    model = model.eval()

    losses = []
    correct_predictions = 0
    total_processed_examples = 0

    with torch.no_grad():
        for i, batch in enumerate(data_iterator):
            if FAST_DEBUG_PREDICTIONS is True:
                # only for testing purposes
                if i == NUM_FAST_DEBUG_ITER:
                    break

            input_ids = batch.text[0].to(device)
            input_ids_lengths = batch.text[1]
            labels = batch.label.to(device)

            data_size = len(labels)
            total_processed_examples += data_size

            predictions = model(input_ids, input_ids_lengths)
            loss = loss_fn(predictions, labels)

            _, preds = torch.max(predictions, dim=1)

            correct_predictions += torch.sum(preds == labels)
            if args.data_parallel is True:
                loss = loss.mean()

            losses.append(loss.item())

    return correct_predictions.double() / total_processed_examples, np.mean(losses), total_processed_examples


def get_predictions(model, data_iter, dataset_df, device, eval_batch_size, print_progress=False):
    model = model.eval()

    review_texts = []
    predictions_list = []
    prediction_probs = []
    real_values = []

    if eval_batch_size is None:
        logger.info("Batch size not specified for printing info setting it to 32")
        batch_size = 32

    t0 = time.time()

    epoch_time = t0

    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            if FAST_DEBUG_PREDICTIONS is True:
                # only for testing purposes
                if i == NUM_FAST_DEBUG_ITER:
                    break

            if print_progress is True:
                if i % 1 == 0:
                    cur_time = time.time() - t0
                    epoch_time = time.time() - epoch_time
                    print("total time: " + str(cur_time) + "s 10 epochs: " + str(epoch_time) + " s  Predicted: " + str(
                        i * batch_size) + " examples current batch: " + str(i))
                    epoch_time = time.time()

            input_ids = batch.text[0].to(device)
            input_ids_lengths = batch.text[1]
            labels = batch.label.to(device)

            predictions = model(input_ids, input_ids_lengths)

            _, preds = torch.max(predictions, dim=1)
            probs = F.softmax(predictions, dim=1)

            # we have to get the original texts
            texts = get_text_list_by_idx(dataset_df, batch.dataset_indices.numpy())
            review_texts.extend(texts)
            predictions_list.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(labels)

    predictions_list = torch.stack(predictions_list).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions_list, prediction_probs, real_values


def get_text_list_by_idx(data_df, indices):
    texts = []
    for index in indices:
        text = data_df.iloc[index].text
        texts.append(text)

    return texts
