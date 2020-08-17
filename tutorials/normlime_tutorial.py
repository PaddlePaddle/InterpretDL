import glob
import sys
import numpy as np
sys.path.append('..')

from interpretdl.interpreter._normlime_base import NormLIMEBase
from interpretdl import LIMECVInterpreter, LIMENLPInterpreter

from assets.resnet import ResNet50


def normlime_example():
    def paddle_model(image_input):
        import paddle.fluid as fluid
        class_num = 1000
        model = ResNet50()
        logits = model.net(input=image_input, class_dim=class_num)
        probs = fluid.layers.softmax(logits, axis=-1)
        return probs

    # The model can be downloaded from
    # http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar
    # More pretrained models can be found in
    # https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleCV/image_classification
    lime = LIMECVInterpreter(paddle_model, "assets/ResNet50_pretrained")
    lime._paddle_prepare()

    # 10 images are used here for example, but more images should be used.
    dataset_dir = "assets"
    image_paths = sorted(glob.glob(dataset_dir + "/*.png"))
    image_paths = image_paths[:10]

    normlime = NormLIMEBase(image_paths, lime.predict_fn)

    # this can be very slow.
    lime_weights = normlime.compute_normlime()


def nlp_example(dataset=False):
    from assets.bilstm import bilstm_net
    import io
    import paddle.fluid as fluid

    def load_vocab(file_path):
        vocab = {}
        with io.open(file_path, 'r', encoding='utf8') as f:
            wid = 0
            for line in f:
                if line.strip() not in vocab:
                    vocab[line.strip()] = wid
                    wid += 1
        vocab["<unk>"] = len(vocab)
        return vocab

    DICT_DIM = 1256606

    def paddle_model(data):
        probs = bilstm_net(data, None, None, DICT_DIM, is_prediction=True)
        return probs

    word_dict = load_vocab("assets/senta_model/bilstm_model/word_dict.txt")
    unk_id = word_dict[""]  #["<unk>"]
    lime = LIMENLPInterpreter(paddle_model,
                              "assets/senta_model/bilstm_model/params")
    lime._paddle_prepare()

    if dataset:
        pad_id = 0
        all_data = []
        max_len = 512
        with io.open(
                "assets/senta_data/test.tsv", "r", encoding='utf8') as fin:
            for line in fin:
                if line.startswith('text_a'):
                    continue
                cols = line.strip().split("\t")
                if len(cols) != 2:
                    sys.stderr.write("[NOTICE] Error Format Line!")
                    continue
                wids = [
                    word_dict[x] if x in word_dict else unk_id
                    for x in cols[0][:200].split(" ")
                ]
                seq_len = len(wids)
                if seq_len < max_len:
                    wids = wids[:max_len]
                all_data.append(wids)
        print('total of %d sentences' % len(all_data))
        lod = all_data
    else:
        reviews = [[
            '交通', '方便', '；', '环境', '很好', '；', '服务态度', '很好', '', '', '房间', '较小'
        ], ['交通', '一般', '；', '环境', '很差', '；', '服务态度', '很差', '房间', '较小']]

        lod = []
        for c in reviews:
            lod.append([word_dict.get(words, unk_id) for words in c])

    base_shape = [[len(c) for c in lod]]
    lod = np.array(sum(lod, []), dtype=np.int64)
    data = fluid.create_lod_tensor(lod, base_shape, fluid.CPUPlace())

    normlime = NormLIMEBase(
        data,
        lime.predict_fn,
        temp_data_file='all_lime_weights_nlp.npz',
        batch_size=50,
        num_samples=500)

    normlime_weights = normlime.compute_normlime_text(unk_id=unk_id)

    #print(normlime_weights)
    id2word = dict(zip(word_dict.values(), word_dict.keys()))
    for label in normlime_weights:
        print(label)
        temp = {
            id2word[wid]: normlime_weights[label][wid]
            for wid in normlime_weights[label]
        }
        W = [(word, weight[0], weight[1]) for word, weight in temp.items()]
        print(sorted(W, key=lambda x: -x[1])[:15])


if __name__ == '__main__':
    targets = sys.argv[1:]
    if 'nlp_mini' in targets:
        nlp_example()
    elif 'nlp_dataset' in targets:
        nlp_example(dataset=True)
    else:
        normlime_example()
