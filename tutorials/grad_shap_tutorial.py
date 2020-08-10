from assets.resnet import ResNet50
from assets.bilstm import bilstm_net_emb
import paddle.fluid as fluid
import numpy as np
import sys
sys.path.append('..')
from interpretdl.interpreter.gradient_shap import GradShapInterpreter
from interpretdl.data_processor.readers import preprocess_image, read_image
from interpretdl.data_processor.visualizer import visualize_grayscale
from PIL import Image
import cv2


def grad_shap_example():
    def predict_fn(data):

        class_num = 1000
        model = ResNet50()
        logits = model.net(input=data, class_dim=class_num)

        probs = fluid.layers.softmax(logits, axis=-1)
        return probs

    img_path = 'assets/catdog.png'
    gs = GradShapInterpreter(predict_fn, "assets/ResNet50_pretrained",
                             True)
    gradients = gs.interpret(
        img_path,
        label=None,
        noise_amount=0.1,
        n_samples=5,
        visual=True,
        save_path='grad_shap_test.jpg')
    
def nlp_example():
    #Dataset: https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
    #Pretrained Model: https://baidu-nlp.bj.bcebos.com/sentiment_classification-1.0.0.tar.gz
    import io
    def load_vocab(file_path):
        """
        load the given vocabulary
        """
        vocab = {}
        with io.open(file_path, 'r', encoding='utf8') as f:
            wid = 0
            for line in f:
                if line.strip() not in vocab:
                    vocab[line.strip()] = wid
                    wid += 1
        vocab["<unk>"] = len(vocab)
        return vocab
    
    def paddle_model(data, alpha, std):
        dict_dim = 1256606
        emb_dim = 128
        # embedding layer
        emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
        emb += fluid.layers.gaussian_random(fluid.layers.shape(emb), std=std)
        emb *= alpha
        probs = bilstm_net_emb(emb, None, None, dict_dim, is_prediction=True)
        return emb, probs
    
    gs = GradShapInterpreter(
        paddle_model,
        "assets/senta_model/bilstm_model/params",
        True,
        model_input_shape=None)
    
    word_dict = load_vocab("assets/senta_model/bilstm_model/word_dict.txt")    
    unk_id=word_dict["<unk>"]
    reviews = [['交通', '方便', '；', '环境', '很好', '；', '服务态度', '很好', '', '', '房间', '较小'],
              ['交通', '方便', '；', '环境', '很好', '；', '服务态度', '很好', '', '', '房间', '较小']]
    
    lod = []
    for c in reviews:
        lod.append([word_dict.get(words, unk_id) for words in c])
    base_shape = [[len(c) for c in lod]]
    lod = np.array(sum(lod, []), dtype=np.int64)
    data = fluid.create_lod_tensor(lod, base_shape, fluid.CPUPlace())
    
    avg_gradients = gs.interpret_text(
        data,
        label=None,
        noise_amount=0.1,
        n_samples=20,
        visual=True,
        save_path=None)
    
    sum_gradients = np.sum(avg_gradients, axis=1).tolist()
    lod = data.lod()
    
    new_array = []
    for i in range(len(lod[0]) - 1):
        new_array.append(sum_gradients[lod[0][i]:lod[0][i + 1]])
    
    print(new_array)
    

if __name__ == '__main__':
    target = sys.argv[1:]
    if 'nlp' in target:
        nlp_example()
    else:
        grad_shap_example()
