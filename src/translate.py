import math
import os
import random
import sys
import time
import datetime
import logging
import argparse

import numpy as np
from six.moves import xrange
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

import data_utils
import network
import _pickle as pickLe


def create_model(source_vocab_size, target_vocab_size, source_vocab_list, target_vocab_list, dropout_rate,
                 max_source_len, max_target_len):
    if args.network == 'seq2seq':
        model = network.Seq2SeqModel(
            source_vocab_size,
            target_vocab_size,
            source_vocab_list,
            target_vocab_list,
            max_source_len,
            max_target_len,
            args.max_depth,
            args.embedding_size,
            args.hidden_size,
            args.num_layers,
            args.max_gradient_norm,
            args.batch_size,
            args.learning_rate,
            dropout_rate)
    elif args.network == 'seq2tree':
        model = network.Seq2TreeModel(
            source_vocab_size,
            target_vocab_size,
            source_vocab_list,
            target_vocab_list,
            max_source_len,
            max_target_len,
            args.max_depth,
            args.embedding_size,
            args.hidden_size,
            args.num_layers,
            args.max_gradient_norm,
            args.batch_size,
            args.learning_rate,
            dropout_rate)
    elif args.network == 'tree2seq':
        model = network.Tree2SeqModel(
            source_vocab_size,
            target_vocab_size,
            source_vocab_list,
            target_vocab_list,
            max_target_len,
            args.max_depth,
            args.embedding_size,
            args.hidden_size,
            args.num_layers,
            args.max_gradient_norm,
            args.batch_size,
            args.learning_rate,
            dropout_rate)
    else:
        model = network.Tree2TreeModel(
            source_vocab_size,
            target_vocab_size,
            source_vocab_list,
            target_vocab_list,
            args.max_depth,
            args.embedding_size,
            args.hidden_size,
            args.num_layers,
            args.max_gradient_norm,
            args.batch_size,
            args.learning_rate,
            dropout_rate,
            args.no_pf,
            args.no_attention)

    if cuda.is_available():
        model.cuda()

    if args.load_model:
        print("Reading model parameters from %s" % args.load_model)
        pretrained_model = torch.load(args.load_model)
        model.load_state_dict(pretrained_model)
    else:
        print("Created model with fresh parameters.")
        model.init_weights(args.param_init)
    return model


def step_seq2seq(model, encoder_inputs, decoder_inputs, source_serialize=True, feed_previous=False):
    if feed_previous == False:
        model.dropout_rate = args.dropout_rate
    else:
        model.dropout_rate = 0.0

    if source_serialize:
        encoder_inputs = Variable(torch.LongTensor(encoder_inputs))
        if cuda.is_available():
            encoder_inputs = encoder_inputs.cuda()

    decoder_inputs = Variable(torch.LongTensor(decoder_inputs))
    if cuda.is_available():
        decoder_inputs = decoder_inputs.cuda()

    predictions = model(encoder_inputs, decoder_inputs, feed_previous=feed_previous)

    if feed_previous:
        output_predictions = []

    total_loss = None
    for time_step in xrange(model.max_target_len - 1):
        prediction = predictions[time_step]
        if feed_previous:
            output_prediction = prediction.max(1)[1]
            output_prediction = output_prediction.squeeze()
            output_predictions.append(output_prediction.data)

        target = decoder_inputs[:, time_step + 1]
        loss = model.loss_function(prediction, target)
        if total_loss is None:
            total_loss = loss
        else:
            total_loss += loss
    total_loss /= len(encoder_inputs)
    if feed_previous == False:
        model.optimizer.zero_grad()
        total_loss.backward()
        if args.max_gradient_norm > 0:
            clip_grad_norm(model.parameters(), args.max_gradient_norm)
        model.optimizer.step()

    if feed_previous:
        return total_loss.item(), output_predictions
    else:
        return total_loss.item()


def step_seq2tree(model, encoder_inputs, init_decoder_inputs, feed_previous=False):
    if feed_previous == False:
        model.dropout_rate = args.dropout_rate
    else:
        model.dropout_rate = 0.0
    encoder_inputs = Variable(torch.LongTensor(encoder_inputs))
    if cuda.is_available():
        encoder_inputs = encoder_inputs.cuda()
    for idx in xrange(len(init_decoder_inputs)):
        init_decoder_inputs[idx] = model.convert_node_to_tensor(init_decoder_inputs[idx])
    predictions_per_batch, prediction_nodes, raw_predictions = model(encoder_inputs, init_decoder_inputs,
                                                                     feed_previous=feed_previous)

    total_loss = None
    for (predictions, decoder_inputs) in predictions_per_batch:
        for time_step in xrange(model.max_target_len - 1):
            prediction = predictions[time_step]
            target = decoder_inputs[:, time_step + 1]
            loss = model.loss_function(prediction, target)
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
    total_loss /= len(encoder_inputs)

    if feed_previous:
        output_predictions = []

        for node_idx in raw_predictions:
            output_predictions.append(model.tree2seq(prediction_nodes, node_idx))

    if feed_previous == False:
        model.optimizer.zero_grad()
        total_loss.backward()
        if args.max_gradient_norm > 0:
            clip_grad_norm(model.parameters(), args.max_gradient_norm)
        model.optimizer.step()

    if feed_previous:
        return total_loss.item(), output_predictions
    else:
        return total_loss.item()


def step_tree2tree(model, encoder_inputs, init_decoder_inputs, feed_previous=False):
    if feed_previous == False:
        model.dropout_rate = args.dropout_rate
    else:
        model.dropout_rate = 0.0

    predictions_per_batch, prediction_managers = model(encoder_inputs, init_decoder_inputs, feed_previous=feed_previous)

    total_loss = None
    for (predictions, target) in predictions_per_batch:
        loss = model.loss_function(predictions, target)
        if total_loss is None:
            total_loss = loss
        else:
            total_loss += loss

    total_loss /= len(encoder_inputs)

    if feed_previous:
        output_predictions = []

        for prediction_manager in prediction_managers:
            output_predictions.append(model.tree2seq(prediction_manager, 1))

    if feed_previous == False:
        model.optimizer.zero_grad()
        total_loss.backward()
        if args.max_gradient_norm > 0:
            clip_grad_norm(model.parameters(), args.max_gradient_norm)
        model.optimizer.step()

    for idx in range(len(encoder_inputs)):
        encoder_inputs[idx].clear_states()

    if feed_previous:
        return total_loss.item(), output_predictions
    else:
        return total_loss.item()


def evaluate(model, test_set, source_vocab, target_vocab, source_vocab_list, target_vocab_list):
    test_loss = 0
    acc_tokens = 0
    tot_tokens = 0
    acc_programs = 0
    tot_programs = len(test_set)
    res = []

    for idx in xrange(0, len(test_set), args.batch_size):
        encoder_inputs, decoder_inputs = model.get_batch(test_set, start_idx=idx)
        if args.network == 'seq2seq' or args.network == 'tree2seq':
            if args.network == 'seq2seq':
                source_serialize = True
            else:
                source_serialize = False
            eval_loss, raw_outputs = step_seq2seq(model, encoder_inputs, decoder_inputs,
                                                  source_serialize=source_serialize, feed_previous=True)
        elif args.network == 'seq2tree':
            eval_loss, raw_outputs = step_seq2tree(model, encoder_inputs, decoder_inputs, feed_previous=True)
        else:
            eval_loss, raw_outputs = step_tree2tree(model, encoder_inputs, decoder_inputs, feed_previous=True)
        test_loss += len(encoder_inputs) * eval_loss
        for i in xrange(len(encoder_inputs)):
            if idx + i >= len(test_set):
                break
            current_output = []
            if args.network == 'seq2seq' or args.network == 'tree2seq':
                for j in xrange(model.max_target_len - 1):
                    current_output.append(raw_outputs[j][i])
                    if raw_outputs[j][i] == data_utils.EOS_ID:
                        break
            else:
                for j in xrange(len(raw_outputs[i])):
                    current_output.append(raw_outputs[i][j])

            start_evaluation_datetime = datetime.datetime.now()
            if args.network == 'tree2tree' or args.network == 'tree2seq':
                current_source, current_target, current_source_manager, current_target_manager = test_set[idx + i]
            else:
                current_source, current_target = test_set[idx + i]
            if args.network != 'seq2seq' and args.network != 'tree2seq':
                current_target_print = data_utils.serialize_tree_with_vocabulary(current_target, target_vocab)
                current_target = data_utils.serialize_tree(current_target)
            if args.network == 'tree2tree':
                current_source_print = data_utils.serialize_tree_with_vocabulary(current_source, source_vocab)
                current_source = data_utils.serialize_tree(current_source)

            #print("Evaluation time: %s seconds" % (datetime.datetime.now() - start_evaluation_datetime))
            #print((datetime.datetime.now() - start_evaluation_datetime))
            res.append((current_source, current_target, current_output))
            current_output_print = data_utils.serialize_seq_with_vocabulary(current_output, target_vocab)
            #print("--Current source / Current target / Current output--")
            #print(current_source_print)
            #print(current_target_print)
            #print(current_output_print)
            #print(source_vocab)
            #print("---")
            tot_tokens += len(current_target)
            all_correct = 1
            wrong_tokens = 0
            for j in xrange(len(current_output)):
                if j >= len(current_target):
                    break
                if current_output[j] == current_target[j]:
                    acc_tokens += 1
                else:
                    all_correct = 0
                    wrong_tokens += 1
            acc_programs += all_correct


    test_loss /= tot_programs
    print("  eval: loss %.2f" % test_loss)
    print("  eval: accuracy of tokens %.2f" % (acc_tokens * 1.0 / tot_tokens))
    print("  eval: accuracy of programs %.2f" % (acc_programs * 1.0 / tot_programs))

    print(acc_tokens, tot_tokens, acc_programs, tot_programs)


def train(train_data, val_data, source_vocab, target_vocab, source_vocab_list, target_vocab_list, source_serialize,
          target_serialize):

    train_model = True;
    build_from_scratch = True;
    pretrained_model_path = "neuralnetwork.pth";
    if (train_model):

        print ("Reading training and val data :")
        train_set = data_utils.prepare_data(train_data, source_vocab, target_vocab, args.input_format,
                                            args.output_format,
                                            source_serialize, target_serialize)

        val_set = data_utils.prepare_data(val_data, source_vocab, target_vocab, args.input_format, args.output_format,
                                          source_serialize, target_serialize)

        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)

        start_time = time.time()
        start_datetime = datetime.datetime.now()

        if (build_from_scratch):
            print("Creating %d layers of %d units." % (args.num_layers, args.hidden_size))
            model = create_model(len(source_vocab), len(target_vocab), source_vocab_list, target_vocab_list, args.dropout_rate,
                                 args.max_source_len, args.max_target_len)
        else:
            print("Loading the pretrained model")
            model = torch.load(pretrained_model_path)

        print("Training model")
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        train_data_size = len(train_set)

        total_progress = args.num_epochs * (train_data_size/args.batch_size + 1)

        for epoch in range(args.num_epochs):
            random.shuffle(train_set)
            for batch_idx in range(0, train_data_size, args.batch_size):
                start_time = time.time()
                encoder_inputs, decoder_inputs = model.get_batch(train_set, start_idx=batch_idx)
                print("train_set %s" % train_set)
                print("%s <-- encoder_inputs" % encoder_inputs[0].trees[0].root)

                if args.network == 'seq2seq' or args.network == 'tree2seq':
                    step_loss = step_seq2seq(model, encoder_inputs, decoder_inputs, source_serialize=source_serialize,
                                             feed_previous=False)
                elif args.network == 'seq2tree':
                    step_loss = step_seq2tree(model, encoder_inputs, decoder_inputs, feed_previous=False)
                else:
                    step_loss = step_tree2tree(model, encoder_inputs, decoder_inputs, feed_previous=False)


                print("%s <-- decoder_inputs" % decoder_inputs[0].trees[0].root)


                step_time += (time.time() - start_time) / args.steps_per_checkpoint
                loss += step_loss / args.steps_per_checkpoint
                current_step += 1

                print("%s/" % current_step + "%s" % total_progress )

                if current_step % args.learning_rate_decay_steps == 0 and model.learning_rate > 0.0001:
                    model.decay_learning_rate(args.learning_rate_decay_factor)

                if current_step % args.steps_per_checkpoint == 0:
                    print ("learning rate %.4f step-time %.2f loss "
                           "%.2f" % (model.learning_rate, step_time, loss))
                    previous_losses.append(loss)
                    ckpt_path = os.path.join(args.train_dir, "translate_" + str(current_step) + ".ckpt")
                    ckpt = model.state_dict()
                    torch.save(ckpt, ckpt_path)
                    step_time, loss = 0.0, 0.0

                    encoder_inputs, decoder_inputs = model.get_batch(val_set, start_idx=0)
                    if args.network == 'seq2seq' or args.network == 'tree2seq':
                        eval_loss, decoder_outputs = step_seq2seq(model, encoder_inputs, decoder_inputs,
                                                                  source_serialize=source_serialize, feed_previous=True)
                    elif args.network == 'seq2tree':
                        eval_loss, decoder_outputs = step_seq2tree(model, encoder_inputs, decoder_inputs,
                                                                   feed_previous=True)
                    else:
                        eval_loss, decoder_outputs = step_tree2tree(model, encoder_inputs, decoder_inputs,
                                                                    feed_previous=True)
                    print("  eval: loss %.2f" % eval_loss)
                    sys.stdout.flush()
        print("Training time: %s seconds" % (datetime.datetime.now() - start_datetime))
        print("Saving model")
        torch.save(model, "neuralnetwork.pth")
    else : # not train_model
        print("Loading the pretrained model")
        model = torch.load(pretrained_model_path)

    print("Evaluating model")
    start_evaluation_datetime = datetime.datetime.now()
    test_data = json.load(open(args.test_data, 'r'))
    test_set = data_utils.prepare_data(test_data, source_vocab, target_vocab, args.input_format, args.output_format,
                                       source_serialize, target_serialize)
    evaluate(model, test_set, source_vocab, target_vocab, source_vocab_list, target_vocab_list)
    print("Total Evaluation time: %s seconds" % (datetime.datetime.now() - start_evaluation_datetime))

def test(test_data, source_vocab, target_vocab, source_vocab_list, target_vocab_list, source_serialize,
         target_serialize):
    model = create_model(len(source_vocab), len(target_vocab), source_vocab_list, target_vocab_list, 0.0,
                         args.max_source_len, args.max_target_len)
    test_set = data_utils.prepare_data(test_data, source_vocab, target_vocab, args.input_format, args.output_format,
                                       source_serialize, target_serialize)
    evaluate(model, test_set, source_vocab, target_vocab, source_vocab_list, target_vocab_list)


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default='tree2tree',
                    choices=['seq2seq', 'seq2tree', 'tree2seq', 'tree2tree'])
parser.add_argument('--param_init', type=float, default=0.1,
                    help='Parameters are initialized over uniform distribution in (-param_init, param_init)')
parser.add_argument('--num_epochs', type=int, default=10, help='number of training epochs') #default 30
parser.add_argument('--learning_rate', type=float, default=0.01, # default 0.005
                    help='learning rate')
parser.add_argument('--learning_rate_decay_factor', type=float, default=0.8,
                    help='learning rate decays by this much')
parser.add_argument('--learning_rate_decay_steps', type=int, default=2000,
                    help='decay the learning rate after certain steps')
parser.add_argument('--max_gradient_norm', type=float, default=5.0,
                    help='clip gradients to this norm')
parser.add_argument('--batch_size', type=int, default=100, #default 100
                    help='batch size')
parser.add_argument('--max_depth', type=int, default=100,
                    help='max depth for tree models')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='size of each model layer')
parser.add_argument('--embedding_size', type=int, default=256,
                    help='size of the embedding')
parser.add_argument('--dropout_rate', type=float, default=0.5,
                    help='dropout rate')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the model')
parser.add_argument('--source_vocab_size', type=int, default=0,
                    help='source vocabulary size (0: no limit)')
parser.add_argument('--target_vocab_size', type=int, default=0,
                    help='target vocabulary size (0: no limit)')
parser.add_argument('--train_dir', type=str, default='../model_ckpts/tree2tree/',
                    help='training directory')
parser.add_argument('--train_data', type=str, default='../data/CS-JS/BL/preprocessed_progs_train.json',
                    help='training data')
parser.add_argument('--val_data', type=str, default='../data/CS-JS/BL/preprocessed_progs_valid.json',
                    help='training data')
parser.add_argument('--test_data', type=str, default='../data/CS-JS/BL/preprocessed_progs_test.json',
                    help='test data')
parser.add_argument('--load_model', type=str, default=None,
                    help='path to the pretrained model')
parser.add_argument('--vocab_filename', type=str, default=None,
                    help='filename for the vocabularies')
parser.add_argument('--steps_per_checkpoint', type=int, default=500,
                    help='number of training steps per checkpoint')
parser.add_argument('--max_source_len', type=int, default=115,
                    help='max length for input')
parser.add_argument('--max_target_len', type=int, default=315,
                    help='max length for output')
parser.add_argument('--test', action='store_true', help='set to true for testing')
parser.add_argument('--input_format', type=str, default='tree', choices=['seq', 'tree'])
parser.add_argument('--output_format', type=str, default='tree', choices=['seq', 'tree'])
parser.add_argument('--no_attention', action='store_true', help='set to true to disable attention')
parser.add_argument('--no_pf', action='store_true', help='set to true to disable parent attention feeding')
args = parser.parse_args()


def main():
    if args.network == 'seq2seq' or args.network == 'seq2tree':
        source_serialize = True
    else:
        source_serialize = False
    if args.network == 'seq2seq' or args.network == 'tree2seq':
        target_serialize = True
    else:
        target_serialize = False
    if args.no_attention:
        args.no_pf = True
    train_data = json.load(open(args.train_data, 'r'))
    source_vocab, target_vocab, source_vocab_list, target_vocab_list = data_utils.build_vocab(train_data,
                                                                                              args.vocab_filename,
                                                                                              args.input_format,
                                                                                              args.output_format)
    if args.test:
        test_data = json.load(open(args.test_data, 'r'))
        test(test_data, source_vocab, target_vocab, source_vocab_list, target_vocab_list, source_serialize,
             target_serialize)
    else:
        val_data = json.load(open(args.val_data, 'r'))
        # print("Val data %s" % val_data)
        train(train_data, val_data, source_vocab, target_vocab, source_vocab_list, target_vocab_list, source_serialize,
              target_serialize)


main()