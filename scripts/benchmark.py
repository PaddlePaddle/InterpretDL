import interpretdl as it
from tqdm import tqdm
import argparse
import logging
import os
import json
from glob import glob
from paddle.vision.models import resnet50
from datetime import datetime


def main(args):
    # data
    def sort_key_func(x):
        return x.split('/')[-1]
    files = glob("/root/datasets/ImageNet_org/val/*/*")
    list_image_paths = sorted(files, key=sort_key_func)[:args.num_images]    

    # model 
    if args.model.lower() == 'resnet50':
        if 'lrp' == args.it:
            from tutorials.assets.lrp_model import resnet50_lrp
            paddle_model = resnet50_lrp(pretrained=True)
        else:
            paddle_model = resnet50(pretrained=True)
    else:
        paddle_model = resnet50(pretrained=True)

    # interpreter instance
    to_test_list = {
        'lime': it.LIMECVInterpreter,
        'gradcam': it.GradCAMInterpreter,
        'intgrad': it.IntGradCVInterpreter,
        'smoothgrad': it.SmoothGradInterpreter,
        'gradshap': it.GradShapCVInterpreter,
        'scorecam': it.ScoreCAMInterpreter,
        'glime': None,
        'lrp': it.LRPCVInterpreter
    }
    interpreter = to_test_list[args.it](paddle_model, device=args.device)
    # interpreter configs
    it_configs = args.it_configs
    eval_configs = args.eval_configs

    # evaluator instance
    del_ins_evaluator = it.DeletionInsertion(paddle_model, device=args.device)
    pert_evaluator = it.Perturbation(paddle_model, device=args.device, compute_MoRF=False)
    
    # compute exp
    del_scores = []
    ins_scores = []
    LeRF_scores = []
    for img_path in tqdm(list_image_paths, leave=True, position=0):
        exp = interpreter.interpret(img_path, **it_configs, visual=False)
        if hasattr(interpreter, 'lime_results'):
            exp = interpreter.lime_results
        results = del_ins_evaluator.evaluate(img_path, exp, **eval_configs)
        del_scores.append(results['deletion_score'])
        ins_scores.append(results['insertion_score'])

        # print(results['del_probas'])
        # print(results['ins_probas'])

        results = pert_evaluator.evaluate(img_path, exp, **eval_configs)
        LeRF_scores.append(results['LeRF_score'])

        # print(results['LeRF_probas'])

    print("Del score:\t", sum(del_scores) / len(del_scores))
    print("Ins score:\t", sum(ins_scores) / len(ins_scores))
    print("LeRF score:\t", sum(LeRF_scores) / len(LeRF_scores))

    logging.info(f"Del score:\t {sum(del_scores) / len(del_scores): .5f}")
    logging.info(f"Ins score:\t {sum(ins_scores) / len(ins_scores): .5f}")
    logging.info(f"LeRF score:\t {sum(LeRF_scores) / len(LeRF_scores): .5f}")

    logging.info(f"{sum(del_scores) / len(del_scores): .5f} \t {sum(ins_scores) / len(ins_scores): .5f} \t {sum(LeRF_scores) / len(LeRF_scores): .5f}")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="default", type=str, help="name of experiments")
    parser.add_argument('--model', default="ResNet50", type=str, help="model name")
    parser.add_argument('--it', default="lime", type=str, help="interpreter name")
    parser.add_argument('--it_configs', default='{}', type=json.loads, help="arguments for interpreter")
    parser.add_argument('--eval_configs', default='{}', type=json.loads, help="arguments for evaluator")
    parser.add_argument('--device', default="gpu:0", type=str, help="device")
    parser.add_argument('--num_images', default=50, type=int, help="number of images for evaluation")
    args = parser.parse_args()

    tik = datetime.now()
    os.makedirs('./work_dirs', exist_ok=True)
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(
        filename=f'./work_dirs/{args.name}_{args.model}_{args.it}_{args.num_images}.log', 
        filemode='w', 
        format=FORMAT,
        level=getattr(logging, 'INFO')
    )
    logging.info(f'{args}\n')

    main(args)

    logging.info(f"Time: {datetime.now() - tik} s.")