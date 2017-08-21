import tensorflow as tf
import os
import json
import subprocess
import numpy as np
from scipy.misc import imread, imresize
from scipy import misc
 
from train import build_forward
from classifier import Classifier, normalize_coords
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes
from tensorflow.python.tools import freeze_graph

import cv2
import argparse
import random
from time import time

def get_image_dir(args):
    weights_iteration = int(args.weights.split('-')[-1]) if args.weights else "production"
    results_path = "screens"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    dirname = os.path.dirname(args.weights) if args.weights else "screens"
    expname = '_' + args.expname if args.expname else ''
    image_dir = '%s/images_%s_%s%s' % (dirname, os.path.basename(os.path.dirname(args.datadir)), weights_iteration, expname)
    return image_dir

def load_frozen_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

def get_results(args, H, data_dir):
    tf.reset_default_graph()
    if args.frozen_graph:
        graph = load_frozen_graph(args.graphfile)
    else:
        new_saver = tf.train.import_meta_graph(args.graphfile)
    NUM_THREADS = 8
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS),
            graph=graph if args.frozen_graph else None) as sess:
        sess.run(tf.global_variables_initializer())
        if args.frozen_graph:
            x_in = graph.get_tensor_by_name('x_in:0')
            pred_boxes = graph.get_tensor_by_name('add:0')
            pred_confidences = graph.get_tensor_by_name('Reshape_2:0')
        else:
            new_saver.restore(sess, args.weights)
            x_in = tf.get_collection('placeholders')[0]
            pred_boxes, pred_confidences = tf.get_collection('vars')
            #freeze_graph.freeze_graph("overfeat.pb", "", False, args.weights, "add,Reshape_2", "save/restore_all",
             #"save/Const:0", "overfeat_frozen.pb", False, '') 

        pred_annolist = al.AnnoList()
        
        included_extenstions = ['jpg', 'bmp', 'png', 'gif']
        image_names = [fn for fn in os.listdir(args.datadir) if any(fn.lower().endswith(ext) for ext in included_extenstions)]
        image_dir = get_image_dir(args)
        subprocess.call('mkdir -p %s' % image_dir, shell=True)
        for i in range(len(image_names)):
            image_name = image_names[i]
            if H['grayscale']:
                orig_img = imread('%s/%s' % (data_dir, image_name), mode = 'RGB' if random.random() < H['grayscale_prob'] else 'L')
                if len(orig_img.shape) < 3:
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
            else:
                orig_img = imread('%s/%s' % (data_dir, image_name), mode = 'RGB')
            img = imresize(orig_img, (H["image_height"], H["image_width"]), interp='cubic')
            feed = {x_in: img}
            start_time = time()
            (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
            time_2 = time()
            pred_anno = al.Annotation()
            pred_anno.imageName = image_name
            new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                            use_stitching=True, rnn_len=H['rnn_len'], min_conf=args.min_conf, tau=args.tau, show_suppressed=args.show_suppressed)
            print(time() - start_time)
            pred_anno.rects = rects
            pred_anno.imagePath = os.path.abspath(data_dir)
            pred_anno = rescale_boxes((H["image_height"], H["image_width"]), pred_anno, orig_img.shape[0], orig_img.shape[1], test=True)
            pred_annolist.append(pred_anno)
           
            imname = '%s/%s' % (image_dir, os.path.basename(image_name))
            misc.imsave(imname, new_img)
            if i % 25 == 0:
                print(i)
    return pred_annolist



def misis_forward(annos):
    def sort_fn(x):
        return x.filename()
    annos = list(sorted(annos, key=sort_fn))
    res = np.zeros(len(annos))
    for i, anno in enumerate(annos):
        import ipdb; ipdb.set_trace()
        img = imread(anno.filename())
        for rect in anno.rects:
            x1 = int(rect.x1)
            y1 = int(rect.y1)
            x2 = int(rect.x2)
            y2 = int(rect.y2)
            ans = img[y1:y2, x1:x2]
            if ans:
                res[i] = 1
                break
        res[i] = 0
    ans = np.where(res == 1)
    return ans if ans >= 0 else -1 

def video_results(args, H, filepath):
    tf.reset_default_graph()
    if args.frozen_graph:
        graph = load_frozen_graph(args.graphfile)
    else:
        new_saver = tf.train.import_meta_graph(args.graphfile)
    NUM_THREADS = 8
    classifier_weights = 'output_classifier/classifier_2017_08_09_14.14/save.ckpt-7001'
    classifier = Classifier((256, 256), 2, classifier_weights)
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS),
            graph=graph if args.frozen_graph else None) as sess:
        sess.run(tf.global_variables_initializer())
        if args.frozen_graph:
            x_in = graph.get_tensor_by_name('x_in:0')
            pred_boxes = graph.get_tensor_by_name('add:0')
            pred_confidences = graph.get_tensor_by_name('Reshape_2:0')
        else:
            new_saver.restore(sess, args.weights)
            x_in = tf.get_collection('placeholders')[0]
            pred_boxes, pred_confidences = tf.get_collection('vars')
        #import ipdb; ipdb.set_trace()args
        cap = cv2.VideoCapture(args.datadir)
        print(cap.isOpened())
        #while not cap.isOpened():
            #cap = cv2.VideoCapture("./IMG_2764.MOV")
        
            #print "Wait for the header"
        #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        #out = cv2.VideoWriter('output.avi',fourcc, 20, (640,480))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            img = imresize(frame, (H["image_height"], H["image_width"]), interp='cubic')
            feed = {x_in: img}
            start_time = time()
            (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
            new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                use_stitching=True, rnn_len=H['rnn_len'], min_conf=args.min_conf, tau=args.tau, show_suppressed=args.show_suppressed)
            for rect in rects:
                x1, y1, x2, y2 = normalize_coords(rect.x1, rect.y1, rect.x2, rect.y2, H["image_width"], H["image_height"])
                logits = classifier.run(imresize(img[y1:y2, x1:x2], (256, 256)))
                index = np.argmax(logits)
                cv2.putText(new_img, str(index), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
            time_2 = time()
            print(time_2 - start_time)
            #out.write(new_img)
            cv2.imshow('frame',new_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=False)
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--graphfile', required=True)
    parser.add_argument('--expname', default='')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--logdir', default='output')
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--tau', default=0.25, type=float)
    parser.add_argument('--min_conf', default=0.5, type=float)
    parser.add_argument('--show_suppressed', default=False, type=bool)
    parser.add_argument('--frozen_graph', default=False, type=bool)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    hypes_file = '%s/hypes.json' % os.path.dirname(args.weights) if args.weights else "hypes/lstm_rezoom.json"
    with open(hypes_file, 'r') as f:
        H = json.load(f)
    expname = '_' + args.expname  if args.expname else ''
    
    video_results(args, H, os.path.dirname(args.datadir))

    #pred_annolist = get_results(args, H, os.path.dirname(args.datadir))
    #misis_forward(pred_annolist)