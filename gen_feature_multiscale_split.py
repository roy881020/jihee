import os
import sys
import cPickle as cp
import argparse
import multiprocessing

import numpy as np
import skimage
import skimage.io

CAFFE_ROOT = r'\\msralab\ProjectData\ehealth04\v-jisoh\image\code\caffe_microsoft'
sys.path.insert(0, os.path.join(CAFFE_ROOT, r'Build\x64\Release\pycaffe'))
import caffe

def p_norm(data, p):
    return np.power(np.average(np.power(data, p), axis=0), 1.0/p)

def final_feature(features, feature_type):
    if feature_type == 'avg':
        return np.mean(features, axis=0)
    elif feature_type == 'max':
        return np.max(features, axis=0)
    elif feature_type == 'p2':
        return p_norm(features, 2.0)
    elif feature_type == 'p3':
        return p_norm(features, 3.0)
    elif feature_type == 'p4':
        return p_norm(features, 4.0)
    elif feature_type == 'p5':
        return p_norm(features, 5.0)
    else:
        assert False

def crop_center(image, size):
    h, w, _ = image.shape
    x1 = h / 2 - size / 2
    x2 = x1 + size
    y1 = w / 2 - size / 2
    y2 = y1 + size
    return image[x1:x2, y1:y2, :]

def process_image(classifier, args, image_path):
    roi_size = args.roi_size
    grid_size = args.grid_size
    new_size = args.new_size
    input_size = args.input_size
    stride = (roi_size - new_size) / max(1, grid_size - 1)
    feature_types = args.feature_types.split(',')
    scales = map(lambda x: float(x), args.scales.split(','))

    image = skimage.img_as_float(skimage.io.imread(image_path))

    h, w, _ = image.shape
    assert h == w and h >= roi_size

    image = crop_center(image, roi_size)
    
    sub_rois = []
    for x in xrange(grid_size):
        for y in xrange(grid_size):
            x1 = x * stride
            x2 = x1 + new_size
            y1 = y * stride
            y2 = y1 + new_size
            assert 0 <= x1 and x2 <= roi_size and 0 <= y1 and y2 <= roi_size
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            for scale in scales:
                scaled_size = int(scale * new_size)
                x1 = cx - scaled_size / 2
                y1 = cy - scaled_size / 2
                x2 = x1 + scaled_size
                y2 = y1 + scaled_size
                #print(input_size)
                #print(image[x1:x2, y1:y2, :])

                sub_rois.append(skimage.transform.resize(image[x1:x2, y1:y2, :], (input_size, input_size)))

    output = classifier.predict(sub_rois, oversample=False)

    results = {}
    for feature_type in feature_types:
        results[feature_type] = []
    assert output.shape[0] == grid_size * grid_size * len(scales)
    dim = output.shape[1]
    for j in xrange(0, output.shape[0], len(scales)):
        features = output[j:j+len(scales)].reshape(dim * len(scales), -1).transpose()
        assert features.shape[1] == dim * len(scales)
        for feature_type in feature_types:
            results[feature_type].append(final_feature(features, feature_type))
    #print(results)
    return results

def process_one_part(image_paths, slide_list, gpu_id, modulo, chosen, args):
    mean_values = np.array(map(lambda x: float(x), args.mean_values.split(',')))
    feature_types = args.feature_types.split(',')

    if gpu_id == -1:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()
    classifier = caffe.Classifier(
        args.network_proto, args.model_path,
        channel_swap=(2,1,0),
        mean=mean_values,
        raw_scale=255)
    
    for i in xrange(len(slide_list)):
        if i % modulo == chosen:
            slide_name = slide_list[i]
            results = {}
            for feature_type in feature_types:
                results[feature_type] = []
            for image_path in image_paths[slide_name]:
                result = process_image(classifier, args, os.path.join(args.data_path, image_path))
                for feature_type in feature_types:
                    results[feature_type].extend(result[feature_type])
            for feature_type in feature_types:
                with open(os.path.join(args.save_path, '%s_%s.dump' % (slide_name, feature_type)), 'wb') as f:
                    cp.dump(results[feature_type], f, protocol=cp.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', type=str, help='Path of raw data.')
    parser.add_argument('--roi_size', dest='roi_size', type=int, help='Size of ROI.')
    parser.add_argument('--new_size', dest='new_size', type=int, help='Size of scale 1.')
    parser.add_argument('--grid_size', dest='grid_size', type=int, help='Grid size.')
    parser.add_argument('--input_size', dest='input_size', type=int, help='Size for network input.')
    parser.add_argument('--network_proto', dest='network_proto', type=str, help='Network prototxt file.')
    parser.add_argument('--model_path', dest='model_path', type=str, help='Path to caffemodel file.')
    parser.add_argument('--mean_values', dest='mean_values', type=str, default='104,117,123', help='Mean values of network input.')
    parser.add_argument('--gpu_ids', dest='gpu_ids', type=str, default='-1', help='GPU IDs for feature extraction.')
    parser.add_argument('--save_path', dest='save_path', type=str, help='Save path.')
    parser.add_argument('--feature_types', dest='feature_types', type=str, default='avg,p2,p3,p4,p5,max', help='Feature types to extract.')
    parser.add_argument('--scales', dest='scales', type=str, default='0.25,0.5,1', help='Scales.')
    args = parser.parse_args()

    gpu_ids = map(lambda x: int(x), args.gpu_ids.split(','))
    feature_types = args.feature_types.split(',')

    image_paths = {}
    with open(os.path.join(args.data_path, 'tags.txt')) as f:
        for line in f:
            t = line.strip().split()
            image_path = t[0]
            print(t[0])
            break
            parts = image_path.split('/')
            if not parts[0] in image_paths:
                image_paths[parts[0]] = []
            image_paths[parts[0]].append(image_path)
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    for f in os.listdir(args.save_path):
        if f.endswith('.dump'):
            parts = f.split('.')
            slide_name = parts[0].split('_')[0]
            if slide_name in image_paths:
                image_paths.pop(slide_name)
    slide_list = image_paths.keys()
    slide_list.sort()

    pool = multiprocessing.Pool(len(gpu_ids))
    for i in xrange(len(gpu_ids)):
        pool.apply_async(
            process_one_part, (image_paths, slide_list, gpu_ids[i], len(gpu_ids), i, args))
    pool.close()
    pool.join()
