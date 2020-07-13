import os
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.param_attr import ParamAttr
from interpretdl.data_processor.readers import preprocess_image
from interpretdl.common.file_utils import download_and_decompress, gen_user_home
import os.path as osp


def paddle_get_fc_weights(var_name="fc_0.w_0"):
    fc_weights = fluid.global_scope().find_var(var_name).get_tensor()
    return np.array(fc_weights)


def paddle_resize(extracted_features, outsize):
    resized_features = fluid.layers.resize_bilinear(extracted_features, outsize)
    return resized_features


def get_pre_models():
    root_path = gen_user_home()
    root_path = osp.join(root_path, '.paddlex')
    h_pre_model_path = osp.join(root_path, "pre_models")
    if not osp.exists(h_pre_model_path):
        if not osp.exists(root_path):
            os.makedirs(root_path)
        url = "https://bj.bcebos.com/paddlex/interpret/pre_models.tar.gz"
        download_and_decompress(url, path=root_path)

    return h_pre_model_path, osp.join(h_pre_model_path, "kmeans_model.pkl")


def avg_using_superpixels(features, segments):
    one_list = np.zeros((len(np.unique(segments)), features.shape[2]))
    for x in np.unique(segments):
        one_list[x] = np.mean(features[segments == x], axis=0)

    return one_list


def centroid_using_superpixels(features, segments):
    from skimage.measure import regionprops
    regions = regionprops(segments + 1)
    one_list = np.zeros((len(np.unique(segments)), features.shape[2]))
    for i, r in enumerate(regions):
        one_list[i] = features[int(r.centroid[0] + 0.5), int(r.centroid[1] +
                                                             0.5), :]
    return one_list


def extract_superpixel_features(feature_map, segments):
    from sklearn.preprocessing import normalize
    centroid_feature = centroid_using_superpixels(feature_map, segments)
    avg_feature = avg_using_superpixels(feature_map, segments)
    x = np.concatenate((centroid_feature, avg_feature), axis=-1)
    x = normalize(x)
    return x


class FeatureExtractor(object):
    """
    This is only used for NormLIME related interpreters.
    """
    def __init__(self):
        self.forward_fn = None
        self.h_pre_model_path = None

    def _check_files(self):
        self.h_pre_model_path, _ = get_pre_models()

    def session_prepare(self):
        self._check_files()

        def conv_bn_layer(input,
                          num_filters,
                          filter_size,
                          stride=1,
                          groups=1,
                          act=None,
                          name=None,
                          is_test=True,
                          global_name='for_kmeans_'):
            conv = fluid.layers.conv2d(
                input=input,
                num_filters=num_filters,
                filter_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                act=None,
                param_attr=ParamAttr(name=global_name + name + "_weights"),
                bias_attr=False,
                name=global_name + name + '.conv2d.output.1')
            if name == "conv1":
                bn_name = "bn_" + name
            else:
                bn_name = "bn" + name[3:]
            return fluid.layers.batch_norm(
                input=conv,
                act=act,
                name=global_name + bn_name + '.output.1',
                param_attr=ParamAttr(global_name + bn_name + '_scale'),
                bias_attr=ParamAttr(global_name + bn_name + '_offset'),
                moving_mean_name=global_name + bn_name + '_mean',
                moving_variance_name=global_name + bn_name + '_variance',
                use_global_stats=is_test
            )

        startup_prog = fluid.Program()
        prog = fluid.Program()
        with fluid.program_guard(prog, startup_prog):
            with fluid.unique_name.guard():
                image_op = fluid.data(name='image', shape=[None, 3, 224, 224], dtype='float32')

                conv = conv_bn_layer(
                    input=image_op,
                    num_filters=32,
                    filter_size=3,
                    stride=2,
                    act='relu',
                    name='conv1_1')
                conv = conv_bn_layer(
                    input=conv,
                    num_filters=32,
                    filter_size=3,
                    stride=1,
                    act='relu',
                    name='conv1_2')
                conv = conv_bn_layer(
                    input=conv,
                    num_filters=64,
                    filter_size=3,
                    stride=1,
                    act='relu',
                    name='conv1_3')
                extracted_features = conv
                resized_features = fluid.layers.resize_bilinear(extracted_features, image_op.shape[2:])

                prog = prog.clone(for_test=True)

        gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
        place = fluid.CUDAPlace(gpu_id)
        # place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        fluid.io.load_persistables(exe, self.h_pre_model_path, prog)

        def forward_fn(data_content):
            images = preprocess_image(data_content)  # transpose to [N, 3, H, W], scaled to [0.0, 1.0]
            result = exe.run(prog, fetch_list=[resized_features], feed={'image': images})
            return result[0][0]

        self.output = resized_features
        self.forward_fn = forward_fn

    def forward(self, data_content):
        if self.forward_fn is None:
            self.session_prepare()

        return self.forward_fn(data_content)