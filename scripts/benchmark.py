
import paddle
from paddle.vision.models import resnet50, resnet101
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

def main(args):
    # data
    def sort_key_func(x):
        # for imagenet val set.
        return x.split('/')[-1]
    
    if '*' in args.data_list:
        data_list = args.data_list.replace('\\', '')
        files = glob(data_list)
        np.random.seed(0)
        files = np.random.permutation(files)
        list_image_paths = files[:args.num_images]
        
    print(args.data_list)
    print(len(list_image_paths))

    # model
    model_init_args = {'pretrained': True, 'num_classes': args.num_classes}
    if args.model_weights is not None:
        model_init_args['pretrained'] = False

    if args.model.lower() == 'resnet50':
        if 'lrp' == args.it:
            from tutorials.assets.lrp_model import resnet50_lrp
            paddle_model = resnet50_lrp(**model_init_args)
        else:
            paddle_model = resnet50(**model_init_args)
    elif args.model.lower() == 'resnet101':
        paddle_model = resnet101(**model_init_args)
    else:
        paddle_model = resnet50(**model_init_args)
    
    ## load weights if given
    if args.model_weights is not None:
        state_dict = paddle.load(args.model_weights)
        paddle_model.set_dict(state_dict)
        print("Load weights from", args.model_weights)

    # interpreter instance
    to_test_list = {
        'lime': it.LIMECVInterpreter,
        'gradcam': it.GradCAMInterpreter,
        'intgrad': it.IntGradCVInterpreter,
        'smoothgrad': it.SmoothGradInterpreter,
        'gradshap': it.GradShapCVInterpreter,
        'scorecam': it.ScoreCAMInterpreter,
        'glime': it.GLIMECVInterpreter,
        'lrp': it.LRPCVInterpreter
    }
    interpreter = to_test_list[args.it](paddle_model, device=args.device)
    # interpreter configs
    it_configs = args.it_configs
    eval_configs = args.eval_configs

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
    pert_evaluator = it.Perturbation(paddle_model, device=args.device, compute_MoRF=False)
    
    # compute exp
    del_scores = []
    ins_scores = []
    LeRF_scores = []
    
    eval_results = {}
    i = 1
    if os.path.exists(f'./work_dirs/{get_exp_id(args)}.npz'):
        eval_results = dict(np.load(f'./work_dirs/{get_exp_id(args)}.npz', allow_pickle=True))

    for img_path in tqdm(list_image_paths, leave=True, position=0):
        if args.it == 'lime' or args.it == 'glime':
            if img_path in eval_results:
                exp = eval_results[img_path].item()
            else:
                exp = interpreter.interpret(img_path, **it_configs, resize_to=256, crop_to=224, visual=False)
                if hasattr(interpreter, 'lime_results'):
                    exp = interpreter.lime_results
        else:
            exp = interpreter.interpret(img_path, **it_configs, resize_to=256, crop_to=224, visual=False)

        if img_path in num_limit_adapter:
            eval_configs['limit_number_generated_samples'] = num_limit_adapter[img_path]
            print(img_path, 'update eval_configs:', eval_configs)
        
        results = del_ins_evaluator.evaluate(img_path, exp, **eval_configs, resize_to=256, crop_to=224)
        del_scores.append(results['del_probas'][:30].mean())
        ins_scores.append(results['ins_probas'][:30].mean())

        # print(results['del_probas'])
        # print(results['ins_probas'])

        # results = pert_evaluator.evaluate(img_path, exp, **eval_configs)
        # LeRF_scores.append(results['LeRF_score'])

        # print(results['LeRF_probas'])
        
        if args.it == 'lime' or args.it == 'glime':
            exp['del_probas'] = results['del_probas']
            exp['ins_probas'] = results['ins_probas']
            eval_results[img_path] = copy.deepcopy(exp)
            np.savez(f'./work_dirs/{get_exp_id(args)}.npz', **eval_results)
        else:
            eval_results[img_path] = {'del_probas': results['del_probas'], 'ins_probas': results['ins_probas']}
            np.savez(f'./work_dirs/{get_exp_id(args)}.npz', **eval_results)
            
        if i % 20 == 0:
            print("Del score:\t", sum(del_scores) / len(del_scores))
            print("Ins score:\t", sum(ins_scores) / len(ins_scores))
            logging.info(f"{i}")
            logging.info(f"Del score:\t {sum(del_scores) / len(del_scores): .5f}")
            logging.info(f"Ins score:\t {sum(ins_scores) / len(ins_scores): .5f}")            
        i += 1

    print("Del score:\t", sum(del_scores) / len(del_scores))
    print("Ins score:\t", sum(ins_scores) / len(ins_scores))
    # print("LeRF score:\t", sum(LeRF_scores) / len(LeRF_scores))

    logging.info(f"Del score:\t {sum(del_scores) / len(del_scores): .5f}")
    logging.info(f"Ins score:\t {sum(ins_scores) / len(ins_scores): .5f}")
    # logging.info(f"LeRF score:\t {sum(LeRF_scores) / len(LeRF_scores): .5f}")

    logging.info(f"{sum(del_scores) / len(del_scores): .5f} \t {sum(ins_scores) / len(ins_scores): .5f}")
    # logging.info(f"{sum(del_scores) / len(del_scores): .5f} \t {sum(ins_scores) / len(ins_scores): .5f} \t {sum(LeRF_scores) / len(LeRF_scores): .5f}")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="default", type=str, help="name of experiments")
    parser.add_argument('--model', default="ResNet50", type=str, help="model name")
    parser.add_argument('--model_weights', default=None, type=str, help="trained model path")
    parser.add_argument('--it', default="lime", type=str, help="interpreter name")
    parser.add_argument('--it_configs', default='{}', type=json.loads, help="arguments for interpreter")
    parser.add_argument('--eval_configs', default='{}', type=json.loads, help="arguments for evaluator")
    parser.add_argument('--device', default="gpu:0", type=str, help="device")
    parser.add_argument('--data_list', default="/root/datasets/ImageNet_org/val/*/*", type=str, help="data_list")
    parser.add_argument('--num_images', default=50, type=int, help="number of images for evaluation")
    parser.add_argument('--num_classes', default=1000, type=int, help="number of classes")
    parser.add_argument(
        '--eval_num_limit_adapter', 
        default=None, 
        type=str, 
        help="arguments for evaluator"
    )
    # glime
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