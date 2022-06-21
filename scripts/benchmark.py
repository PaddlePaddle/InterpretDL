
import paddle
from paddle.vision.models import resnet50, resnet101, vgg16
import interpretdl as it

from tqdm import tqdm
from glob import glob
import numpy as np
import copy
import argparse
import logging
import os
import json
from datetime import datetime


def get_exp_id(args):
    return f'{args.name}_{args.model}_{args.it}_{args.num_images}'

def get_data(args):
    if '*' in args.data_list:
        data_list = args.data_list.replace('\\', '')
        files = glob(data_list)
        if 'ILSVRC2012_val' in files[0]:
            files = sorted(files, key=lambda s: s[-10:])
        else:
            np.random.seed(0)
            files = np.random.permutation(files)
        list_image_paths = files[:args.num_images]
    elif '.txt' in args.data_list:
        list_image_paths = []
        with open(args.data_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                list_image_paths.append(line.strip())
        list_image_paths = list_image_paths[:args.num_images]
    else:
        raise NotImplementedError

    return list_image_paths

def get_model(args):
    model_init_args = {'pretrained': True, 'num_classes': args.num_classes}
    if args.model_weights is not None:
        model_init_args['pretrained'] = False

    models_dict = {
        'resnet50': resnet50,
        'resnet101': resnet101,
        'vgg16': vgg16,
        'vit': None  # to add.
    }
    if 'lrp' == args.it:
        from tutorials.assets.lrp_model import resnet50 as resnet50_lrp
        from tutorials.assets.lrp_model import resnet101 as resnet101_lrp
        from tutorials.assets.lrp_model import vgg16 as vgg16_lrp
        
        models_dict = {
            'resnet50': resnet50_lrp,
            'resnet101': resnet101_lrp,
            'vgg16': vgg16_lrp
        }
        assert args.model.lower() in models_dict, "LRP supports resnet and vgg only."

    paddle_model = models_dict[args.model.lower()](**model_init_args)
    
    ## load weights if given
    if args.model_weights is not None:
        state_dict = paddle.load(args.model_weights)
        paddle_model.set_dict(state_dict)
        print("Load weights from", args.model_weights)
    return paddle_model

def main(args):
    # get data
    list_image_paths = get_data(args)
    print(args.data_list)
    print(len(list_image_paths))

    # get model
    paddle_model = get_model(args)

    # get interpreter instance
    interpreters_dict = {
        'lime': it.LIMECVInterpreter,
        'gradcam': it.GradCAMInterpreter,
        'intgrad': it.IntGradCVInterpreter,
        'smoothgrad': it.SmoothGradInterpreter,
        'gradshap': it.GradShapCVInterpreter,
        'scorecam': it.ScoreCAMInterpreter,
        'glime': it.GLIMECVInterpreter,
        'lrp': it.LRPCVInterpreter
    }
    interpreter = interpreters_dict[args.it](paddle_model, device=args.device)
    
    # interpreter configs
    it_configs = args.it_configs
    # evaluation configs
    eval_configs = args.eval_configs

    # image resize config.
    # depreciated set: {"resize_to": 256, "crop_to": 224}
    img_resize_configs = args.img_resize_configs
    if img_resize_configs is None:
        img_resize_configs = {"resize_to": 224, "crop_to": 224}

    if 'glime' == args.it:
        interpreter.set_global_weights(args.global_weights)

    num_limit_adapter = {}
    if args.eval_num_limit_adapter is not None:
        lime_results = dict(np.load(args.eval_num_limit_adapter, allow_pickle=True))
        img_path_list = list(lime_results.keys())
        for i in range(len(img_path_list)):
            img_path = img_path_list[i]
            b = lime_results[img_path].item()
            num_limit_adapter[img_path] = len(np.unique(b['segmentation']))

    # evaluator instance
    del_ins_evaluator = it.DeletionInsertion(paddle_model, device=args.device)
    
    # compute exp
    del_scores = []
    ins_scores = []
    
    eval_results = {}
    i = 1
    if os.path.exists(f'./work_dirs/{get_exp_id(args)}.npz'):
        logging.info(f"Loading computed results from ./work_dirs/{get_exp_id(args)}.npz")
        eval_results = dict(np.load(f'./work_dirs/{get_exp_id(args)}.npz', allow_pickle=True))

    for img_path in tqdm(list_image_paths, leave=True, position=0):
        if img_path in eval_results:
            # load computed exp.
            eval_result = eval_results[img_path].item()
        else:
            # compute exp. lime_results or array_exp.
            exp = interpreter.interpret(img_path, **it_configs, **img_resize_configs, visual=False)
            if hasattr(interpreter, 'lime_results'):
                exp = interpreter.lime_results

            if img_path in num_limit_adapter:
                eval_configs['limit_number_generated_samples'] = num_limit_adapter[img_path]
                print(img_path, 'update eval_configs:', eval_configs)
            
            # evaluate.
            # eval_result: A dict containing 'deletion_score', 'del_probas', 'deletion_images', 'insertion_score', 
            # 'ins_probas' and 'insertion_images', if compute_deletion and compute_insertion are both True.
            eval_result = del_ins_evaluator.evaluate(img_path, exp, **eval_configs, **img_resize_configs)

            if args.save_eval_result:
                eval_result_to_save = {
                    'del_probas': eval_result['del_probas'], 
                    'ins_probas': eval_result['ins_probas'], 
                    'exp': exp  # lime_results or array_exp.
                }
                eval_results[img_path] = copy.deepcopy(eval_result_to_save)
                np.savez(f'./work_dirs/{get_exp_id(args)}.npz', **eval_results)

        del_scores.append(eval_result['del_probas'].mean())
        ins_scores.append(eval_result['ins_probas'].mean())

        if i % 20 == 0:
            print("Del score:\t", sum(del_scores) / len(del_scores))
            print("Ins score:\t", sum(ins_scores) / len(ins_scores))
            logging.info(f"{i}")
            logging.info(f"Del score:\t {sum(del_scores) / len(del_scores): .5f}")
            logging.info(f"Ins score:\t {sum(ins_scores) / len(ins_scores): .5f}")            
        i += 1

    print("Del score:\t", sum(del_scores) / len(del_scores))
    print("Ins score:\t", sum(ins_scores) / len(ins_scores))

    logging.info(f"Del score:\t {sum(del_scores) / len(del_scores): .5f}")
    logging.info(f"Ins score:\t {sum(ins_scores) / len(ins_scores): .5f}")

    logging.info(f"{sum(del_scores) / len(del_scores): .5f} \t {sum(ins_scores) / len(ins_scores): .5f}")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="default", type=str, help="name of experiments")
    parser.add_argument('--model', default="ResNet50", type=str, help="model name")
    parser.add_argument('--model_weights', default=None, type=str, help="trained model path")
    parser.add_argument('--it', default="lime", type=str, help="interpreter name")
    parser.add_argument('--it_configs', default='{}', type=json.loads, help="arguments for interpreter")
    parser.add_argument('--eval_configs', default='{}', type=json.loads, help="arguments for evaluator")
    parser.add_argument('--img_resize_configs', default=None, type=json.loads, help="arguments for evaluator")
    parser.add_argument('--device', default="gpu:0", type=str, help="device")
    parser.add_argument('--data_list', default="/root/datasets/ImageNet_org/val/*/*", type=str, help="data_list")
    parser.add_argument('--num_images', default=50, type=int, help="number of images for evaluation")
    parser.add_argument('--num_classes', default=1000, type=int, help="number of classes")
    parser.add_argument('--eval_num_limit_adapter', default=None, type=str, help="arguments for evaluator")
    parser.add_argument('--save_eval_result', default=0, type=int, help="save explanations")
    # used for glime only.
    parser.add_argument('--global_weights', default=None, type=str, help="./work_dirs/global_weights_normlime.npy")
    args = parser.parse_args()

    tik = datetime.now()
    os.makedirs('./work_dirs', exist_ok=True)
    FORMAT = '%(asctime)-15s %(message)s'
    
    logging.basicConfig(
        filename=f'./work_dirs/{get_exp_id(args)}.log', 
        filemode='w', 
        format=FORMAT,
        level=getattr(logging, 'INFO')
    )
    logging.info(f'{args}\n')
    print(args)

    main(args)

    logging.info(f"Time: {datetime.now() - tik} s.")